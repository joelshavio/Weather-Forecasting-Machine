import mysql.connector
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

class WeatherForecaster:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'user': 'weather_user',
            'password': 'weather_pass',
            'database': 'weather_db',
            'auth_plugin': 'mysql_native_password'
        }
        self.connection = None
        self.cursor = None
        
    def connect_to_db(self):
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            print("Successfully connected to the database")
            return True
        except mysql.connector.Error as err:
            print(f"Database connection error: {err}")
            return False
    
    def fetch_weather_data(self):
        query = """SELECT date, temperature, humidity, precipitation, windspeed 
                   FROM weather_data ORDER BY date"""
        try:
            self.cursor.execute(query)
            data = self.cursor.fetchall()
            if len(data) < 10:
                print("Warning: Limited historical data may affect forecast accuracy")
            columns = ['date', 'temperature', 'humidity', 'precipitation', 'windspeed']
            return pd.DataFrame(data, columns=columns)
        except mysql.connector.Error as err:
            print(f"Error fetching data: {err}")
            return None
    
    def preprocess_data(self, df):
        try:
            df['date'] = pd.to_datetime(df['date'])
            df['days'] = (df['date'] - df['date'].min()).dt.days
            df['day_of_year'] = df['date'].dt.dayofyear
            df['month'] = df['date'].dt.month
            
            last_date = df['date'].max()
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 4)]
            future_days = [(date - df['date'].min()).days for date in future_dates]
            
            return df, future_days, future_dates
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return None, None, None
    
    def train_models(self, df):
        models = {}
        features = ['days', 'day_of_year', 'month']
        X = df[features]
        
        for param in ['temperature', 'humidity', 'precipitation', 'windspeed']:
            y = df[param]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            print(f"{param} model MAE: {mae:.2f}")
            
            models[param] = model
        
        return models
    
    def make_forecasts(self, models, future_days, future_dates):
        forecasts = []
        base_date = future_dates[0] - pd.Timedelta(days=future_days[0])
        
        for i, days in enumerate(future_days):
            forecast = {'date': future_dates[i].strftime('%Y-%m-%d')}
            day_of_year = future_dates[i].timetuple().tm_yday
            month = future_dates[i].month
            
            input_data = pd.DataFrame({
                'days': [days],
                'day_of_year': [day_of_year],
                'month': [month]
            })
            
            for param, model in models.items():
                prediction = model.predict(input_data)[0]
                if param == 'temperature':
                    prediction = max(-20, min(50, prediction))
                elif param == 'humidity':
                    prediction = max(0, min(100, prediction))
                elif param in ['precipitation', 'windspeed']:
                    prediction = max(0, prediction)
                forecast[param] = round(float(prediction), 1)
            
            forecasts.append(forecast)
        
        return forecasts
    
    def display_forecasts(self, forecasts):
        print("\nWeather Forecast for Next 3 Days:")
        print("="*60)
        print(f"{'Date':<12} {'Temp (°C)':<12} {'Humidity (%)':<14} {'Rain (mm)':<12} {'Wind (km/h)':<12}")
        print("-"*60)
        for fc in forecasts:
            print(
                f"{fc['date']:<12} "
                f"{fc['temperature']:<12} "
                f"{fc['humidity']:<14} "
                f"{fc['precipitation']:<12} "
                f"{fc['windspeed']:<12}"
            )
        print(f"\nForecast generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    def close_connection(self):
        if self.connection and self.connection.is_connected():
            self.cursor.close()
            self.connection.close()
            print("Database connection closed")

    def run_forecasting(self):
        try:
            if not self.connect_to_db():
                return
                
            weather_df = self.fetch_weather_data()
            if weather_df is None or weather_df.empty:
                return
                
            processed_df, future_days, future_dates = self.preprocess_data(weather_df)
            if processed_df is None:
                return
                
            models = self.train_models(processed_df)
            forecasts = self.make_forecasts(models, future_days, future_dates)
            self.display_forecasts(forecasts)
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            
        finally:
            self.close_connection()

if __name__ == "__main__":
    forecaster = WeatherForecaster()
    forecaster.run_forecasting()
