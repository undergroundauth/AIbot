import MetaTrader5 as mt5
import time
import os
import joblib
import pandas as pd
import requests
import numpy as np
import talib
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
import logging
import json  # For news sentiment caching

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NEWS_API_KEY = "d8bb1988d35a46ea98d7256acff01753"

# Connect to MetaTrader 5
def connect_to_mt5():
    if not mt5.initialize():
        logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
        exit()
    logger.info("✅ Connected to MetaTrader 5!")

connect_to_mt5()

# Check Account Info
account_info = mt5.account_info()
if account_info is None:
    logger.error("Failed to fetch account info.")
else:
    logger.info(f"✅ Account Balance: {account_info.balance}")
    logger.info(f"✅ Account Equity: {account_info.equity}")
    logger.info(f"✅ Free Margin: {account_info.margin_free}")
    logger.info(f"✅ Account Leverage: 1:{account_info.leverage}")

class TradingAI:
    feature_names_lgb = [
        "RSI", "MACD", "ATR", "ADX", "VWAP", "Correlation", 
        "News_Sentiment", "Volatility", "Seasonal_Trend", "Order_Flow"
    ]
    feature_names_gbm = [
        "RSI", "MACD", "ATR", "ADX", "VWAP", "Correlation", 
        "News_Sentiment", "Volatility"
    ]
    def __init__(self):
        self.trade_history_file = os.path.join(os.getcwd(), "trade_history.csv")
        self.ensure_csv_exists()
        self.lgb_model = None
        self.gbm_model = None
        self.pattern_model = None
        self.scaler = MinMaxScaler()
        self.load_models()
        self.load_pattern_model()
        self.daily_trades = 0
        self.daily_profit = 0

    def ensure_csv_exists(self):
        required_columns = [
            "Timestamp", "Order_ID", "Symbol", "Type", "Entry_Price", "Stop_Loss", 
            "Take_Profit", "Pips", "Profit", "RSI", "MACD", "ATR", "ADX", "VWAP", 
            "Correlation", "News_Sentiment", "Volatility", "Seasonal_Trend", "Order_Flow"
        ]
        if not os.path.exists(self.trade_history_file):
            logger.info("🔄 Creating trade history file with required columns...")
            pd.DataFrame(columns=required_columns).to_csv(self.trade_history_file, index=False)
        else:
            trade_data = pd.read_csv(self.trade_history_file)
            missing_columns = [col for col in required_columns if col not in trade_data.columns]
            if missing_columns:
                logger.warning(f"⚠️ Missing columns detected in trade history: {missing_columns}. Fixing...")
                for col in missing_columns:
                    trade_data[col] = np.nan
                trade_data.to_csv(self.trade_history_file, index=False)
                logger.info("✅ trade_history.csv updated with missing columns.")

    def calculate_atr(self, symbol, timeframe, period=14):
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 50)
            if rates is None:
                logger.error(f"❌ Failed to fetch data for {symbol}. MT5 Error: {mt5.last_error()}")
                return None
            if len(rates) < period:
                logger.error(f"❌ Not enough data to calculate ATR for {symbol}. Required: {period}, Available: {len(rates)}")
                return None
            df = pd.DataFrame(rates)
            if df.isnull().values.any():
                logger.error(f"❌ Invalid data (NaN values) for {symbol}.")
                return None
            required_columns = ['high', 'low', 'close']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"❌ Missing required columns in data for {symbol}. Available columns: {df.columns}")
                return None
            df['prev_close'] = df['close'].shift(1)
            df['TR'] = df.apply(lambda row: max(
                row['high'] - row['low'],
                abs(row['high'] - row['prev_close']),
                abs(row['low'] - row['prev_close'])
            ), axis=1)
            atr = df['TR'].rolling(window=period).mean().iloc[-1]
            logger.info(f"✅ ATR calculated for {symbol}: {atr}")
            return atr
        except Exception as e:
            logger.error(f"❌ Unexpected error in calculate_atr for {symbol}: {e}")
            return None

    def calculate_position_size(self, symbol):
        return 5.0  # Fixed, adjust as needed

    def place_trade(self, order_type, symbol):
        try:
            confidence_threshold = 0.7
            features = self.get_features(symbol)
            if features is None:
                logger.error(f"❌ Skipping trade for {symbol} due to insufficient data.")
                return
            if self.lgb_model is None:
                logger.warning("⚠️ LightGBM model is not loaded. Skipping trade.")
                return
            trend_direction = self.ensemble_prediction(features)
            confidence = self.lgb_model.predict(features[self.feature_names_lgb])[0]
            if confidence < confidence_threshold:
                logger.warning(f"⚠️ Skipping trade for {symbol} due to low confidence: {confidence:.2f}")
                return
            if not self.match_pattern(features[self.feature_names_lgb]):
                logger.warning(f"⚠️ Skipping trade for {symbol} due to no matching profitable pattern.")
                return
            atr = self.calculate_atr(symbol, mt5.TIMEFRAME_H1)
            if not atr:
                logger.error(f"❌ Failed to calculate ATR for {symbol}.")
                return
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"❌ Failed to fetch tick data for {symbol}.")
                return
            price = tick.ask if order_type == 0 else tick.bid
            stop_loss = round(price - atr * 2.5, 5) if order_type == 0 else round(price + atr * 2.5, 5)
            take_profit = round(price + atr * 5, 5) if order_type == 0 else round(price - atr * 5, 5)
            risk_reward_ratio = abs(take_profit - price) / abs(price - stop_loss)
            if risk_reward_ratio < 2:
                logger.warning(f"⚠️ Skipping trade for {symbol} due to poor risk-reward ratio: {risk_reward_ratio:.2f}")
                return
            position_size = self.calculate_position_size(symbol)
            logger.info(f"📊 Trade Details for {symbol}:")
            logger.info(f"  - Position Size: {position_size}")
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"❌ Failed to fetch symbol info for {symbol}.")
                return
            margin_required = position_size * symbol_info.margin_initial
            if margin_required > account_info.margin_free:
                logger.error(f"❌ Insufficient margin for {symbol}. Required: {margin_required}, Free: {account_info.margin_free}")
                return
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position_size,
                "type": mt5.ORDER_TYPE_BUY if order_type == 0 else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 10,
                "magic": 123456,
                "comment": "AI Trading Bot",
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Trade Executed: {symbol} | Volume: {position_size} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
                self.daily_trades += 1
                self.daily_profit += (take_profit - price) * position_size if order_type == 0 else (price - take_profit) * position_size
                self.record_trade(symbol, order_type, price, stop_loss, take_profit, position_size, features)
            else:
                logger.error(f"❌ Trade execution failed for {symbol}. Error: {result.comment}")
        except Exception as e:
            logger.error(f"❌ Unexpected error in place_trade for {symbol}: {e}")

    def record_trade(self, symbol, order_type, price, stop_loss, take_profit, position_size, features):
        try:
            feature_row = features[self.feature_names_lgb].iloc[0] if isinstance(features, pd.DataFrame) else [np.nan]*10
            trade_details = {
                "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Order_ID": mt5.positions_total() + 1,
                "Symbol": symbol,
                "Type": "Buy" if order_type == 0 else "Sell",
                "Entry_Price": price,
                "Stop_Loss": stop_loss,
                "Take_Profit": take_profit,
                "Pips": (take_profit - price) * 10000 if order_type == 0 else (price - take_profit) * 10000,
                "Profit": (take_profit - price) * position_size if order_type == 0 else (price - take_profit) * position_size,
                "RSI": feature_row["RSI"],
                "MACD": feature_row["MACD"],
                "ATR": feature_row["ATR"],
                "ADX": feature_row["ADX"],
                "VWAP": feature_row["VWAP"],
                "Correlation": feature_row["Correlation"],
                "News_Sentiment": feature_row["News_Sentiment"],
                "Volatility": feature_row["Volatility"],
                "Seasonal_Trend": feature_row["Seasonal_Trend"],
                "Order_Flow": feature_row["Order_Flow"],
            }
            trade_data = pd.DataFrame([trade_details])
            trade_data.to_csv(self.trade_history_file, mode="a", header=not os.path.exists(self.trade_history_file), index=False)
            logger.info(f"✅ Trade recorded for {symbol} in trade_history.csv.")
        except Exception as e:
            logger.error(f"❌ Failed to record trade for {symbol}: {e}")

    def monitor_open_positions(self):
        try:
            open_positions = mt5.positions_get()
            if open_positions is None:
                logger.error(f"❌ Failed to fetch open positions. MT5 Error: {mt5.last_error()}")
                return
            for position in open_positions:
                symbol = position.symbol
                position_id = position.ticket
                order_type = position.type
                sl = position.sl
                tp = position.tp
                current_price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
                if (order_type == mt5.ORDER_TYPE_BUY and (current_price <= sl or current_price >= tp)) or \
                   (order_type == mt5.ORDER_TYPE_SELL and (current_price >= sl or current_price <= tp)):
                    logger.info(f"🔔 SL/TP hit for {symbol} (Position ID: {position_id}). Closing trade...")
                    self.close_position(position_id, symbol)
        except Exception as e:
            logger.error(f"❌ Unexpected error in monitor_open_positions: {e}")

    def close_position(self, position_id, symbol):
        try:
            pos = mt5.positions_get(ticket=position_id)
            if not pos:
                logger.error(f"❌ No open position with ticket {position_id}")
                return
            position = pos[0]
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position_id,
                "deviation": 10,
                "magic": 123456,
                "comment": "AI Trading Bot - Auto Close",
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Position {position_id} closed successfully.")
            else:
                logger.error(f"❌ Failed to close position {position_id}. Error: {result.comment}")
        except Exception as e:
            logger.error(f"❌ Unexpected error in close_position: {e}")

    def load_models(self):
        logger.info("🔄 Loading AI Models...")
        if os.path.exists("lgb_model.txt"):
            try:
                self.lgb_model = lgb.Booster(model_file="lgb_model.txt")
                logger.info("✅ LightGBM Model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Error loading LightGBM model: {e}")
                self.lgb_model = None
        if os.path.exists("gbm_model.pkl"):
            try:
                self.gbm_model = joblib.load("gbm_model.pkl")
                logger.info("✅ Gradient Boosting model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Error loading Gradient Boosting model: {e}")
                self.gbm_model = None

    def load_pattern_model(self):
        if os.path.exists("pattern_model.pkl"):
            self.pattern_model = joblib.load("pattern_model.pkl")
            logger.info("✅ Pattern recognition model loaded.")
        else:
            self.train_pattern_model()

    def train_pattern_model(self):
        trade_data = pd.read_csv(self.trade_history_file)
        if trade_data.empty:
            logger.warning("⚠️ No trade history available for pattern recognition.")
            return
        profitable_trades = trade_data[trade_data["Profit"] > 0]
        if len(profitable_trades) < 2:
            logger.warning("⚠️ Not enough profitable trades to train pattern model. Skipping training.")
            return
        X = profitable_trades[self.feature_names_lgb].values
        num_clusters = min(5, len(profitable_trades))
        if num_clusters < 2:
            logger.warning("⚠️ Not enough unique trades for clustering. Skipping training.")
            return
        self.pattern_model = KMeans(n_clusters=num_clusters, random_state=42)
        self.pattern_model.fit(X)
        joblib.dump(self.pattern_model, "pattern_model.pkl")
        logger.info(f"✅ Pattern recognition model trained with {num_clusters} clusters and saved.")

    def match_pattern(self, features):
        if self.pattern_model is None:
            return False
        cluster = self.pattern_model.predict(features)
        return cluster in [0, 1]

    def update_trade_rewards(self):
        trade_data = pd.read_csv(self.trade_history_file)
        trade_data['Reward'] = trade_data['Profit'].apply(lambda x: 1 if x > 0 else -1)
        trade_data.to_csv(self.trade_history_file, index=False)
        logger.info("✅ Trade rewards updated based on profitability.")

    def update_lgb_model(self):
        trade_data = pd.read_csv(self.trade_history_file)
        if trade_data.empty:
            logger.warning("⚠️ No trade history available for updating the model.")
            return
        X = trade_data[self.feature_names_lgb].values
        y = (trade_data["Profit"] > 0).astype(int).values
        if self.lgb_model is not None:
            self.lgb_model = lgb.train(
                params={
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9
                },
                train_set=lgb.Dataset(X, label=y),
                init_model=self.lgb_model,
                num_boost_round=50
            )
            self.lgb_model.save_model("lgb_model.txt")
            logger.info("✅ LightGBM Model Updated with New Trade Data!")

    def get_news_sentiment(self, symbol):
        try:
            time.sleep(1)  # Short delay for API etiquette
            cache_file = f"news_cache_{symbol}.json"
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                    if time.time() - cached_data["timestamp"] < 3600:
                        return cached_data["sentiment"]
            response = requests.get(
                f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
            )
            response.raise_for_status()
            news_data = response.json()
            sentiment_score = 0
            for article in news_data["articles"]:
                if "positive" in article["title"].lower():
                    sentiment_score += 1
                elif "negative" in article["title"].lower():
                    sentiment_score -= 1
            sentiment = sentiment_score / len(news_data["articles"]) if news_data["articles"] else 0
            with open(cache_file, "w") as f:
                json.dump({"timestamp": time.time(), "sentiment": sentiment}, f)
            return sentiment
        except Exception as e:
            logger.error(f"❌ Error fetching news sentiment: {e}")
            return 0

    def check_economic_events(self):
        url = f"https://www.alphavantage.co/query?function=ECONOMIC_CALENDAR&apikey={NEWS_API_KEY}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            events = response.json()
            if not events or "data" not in events:
                logger.warning("⚠️ No economic event data found in the API response.")
                return False
            high_impact_events = [event for event in events["data"] if event.get("impact") == "high"]
            if high_impact_events:
                logger.warning("⚠️ High-impact economic events detected. Skipping trades.")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error fetching economic events: {e}")
            return False

    def get_features(self, symbol):
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
        if rates is None or len(rates) < 50:
            logger.error(f"❌ Not enough data to compute features for {symbol}")
            return None
        df = pd.DataFrame(rates)
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['MACD'], _, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['ATR'] = self.calculate_atr(symbol, mt5.TIMEFRAME_M15)
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['VWAP'] = (df['close'] + df['low'] + df['high']) / 3
        df['Correlation'] = df['close'].rolling(window=10).corr(df['open'])
        df['News_Sentiment'] = self.get_news_sentiment(symbol)
        df['Volatility'] = df['ATR'] / df['VWAP']
        df['Seasonal_Trend'] = df['close'].rolling(window=30).mean() - df['close'].rolling(window=90).mean()
        df['Order_Flow'] = df['tick_volume'].diff()
        # Prepare features as DataFrame row, column order matching model
        features_row = df.iloc[-1][self.feature_names_lgb]
        features_df = pd.DataFrame([features_row], columns=self.feature_names_lgb)
        # Check for NaN
        if features_df.isnull().any().any():
            logger.error(f"❌ NaN detected in features for {symbol}")
            return None
        return features_df

    def ensemble_prediction(self, features):
        if self.lgb_model is None or self.gbm_model is None:
            logger.warning("⚠️ One or more models are not loaded. Using random prediction.")
            return np.random.choice([0, 1])
        try:
            features_gbm = features[self.feature_names_gbm]
            lgb_prediction = self.lgb_model.predict(features[self.feature_names_lgb])[0]
            gbm_prediction = self.gbm_model.predict(features_gbm)[0]
            final_prediction = (lgb_prediction + gbm_prediction) / 2
            return 1 if final_prediction > 0.5 else 0
        except Exception as e:
            logger.error(f"❌ Error in ensemble_prediction: {e}")
            return np.random.choice([0, 1])

# Train LightGBM Model if it doesn't exist
def train_lgb_model():
    def balance_dataset(trade_data):
        buy_trades = trade_data[trade_data['Type'] == 'Buy']
        sell_trades = trade_data[trade_data['Type'] == 'Sell']
        min_size = min(len(buy_trades), len(sell_trades))
        balanced_data = pd.concat([buy_trades.sample(min_size, random_state=42), sell_trades.sample(min_size, random_state=42)])
        return balanced_data

    if not os.path.exists("lgb_model.txt"):
        logger.info("🚀 Training LightGBM Model...")
        trade_data = pd.read_csv("trade_history.csv")
        trade_data = balance_dataset(trade_data)
        if trade_data.empty:
            logger.error("❌ No trade history available for training.")
            return
        feature_columns = TradingAI.feature_names_lgb
        X = trade_data[feature_columns].values
        y = (trade_data["Profit"] > 0).astype(int).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)
        model.save_model("lgb_model.txt")
        logger.info("✅ LightGBM Model Training Completed & Saved!")

train_lgb_model()

# Main run loop
bot = TradingAI()
try:
    while bot.daily_trades < 10 and abs(bot.daily_profit) < 100:
        if bot.check_economic_events():
            time.sleep(60 * 60)
            continue
        bot.monitor_open_positions()
        for symbol in ["EURUSD", "USDCHF", "GBPUSD", "USDJPY"]:
            features = bot.get_features(symbol)
            if features is not None:
                trend_direction = bot.ensemble_prediction(features)
                bot.place_trade(trend_direction, symbol)
            else:
                logger.error(f"❌ Skipping trade for {symbol} due to insufficient data.")
        time.sleep(300)
except KeyboardInterrupt:
    logger.info("\n🛑 Stopping the bot safely...")
    mt5.shutdown()
    logger.info("✅ Bot stopped gracefully!")
