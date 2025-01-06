import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.exceptions import V20Error
import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv

class OandaClient:
    def __init__(self):
        # Debug: Show all available secrets
        st.write("Available secrets:", list(st.secrets.keys()))
        
        # Try to get credentials from Streamlit secrets first
        try:
            # Debug: Show raw secrets access
            st.write("Trying to access secrets as dictionary...")
            api_key = st.secrets["OANDA_API_KEY"]  # Direct dictionary access
            self.account_id = st.secrets["OANDA_ACCOUNT_ID"]  # Direct dictionary access
            
            # Debug: Show what we got
            st.write("Raw API Key length:", len(api_key) if api_key else "Not found")
            st.write("Raw Account ID:", self.account_id if self.account_id else "Not found")
            
            # Add 'Bearer ' prefix if not present and if we have an API key
            if api_key:
                self.api_key = f"Bearer {api_key}" if not api_key.startswith('Bearer ') else api_key
                # Debug: Show final API key format
                st.write("Final API Key format:", "Bearer *" * len(api_key.replace('Bearer ', '')))
            else:
                self.api_key = None
                
        except Exception as e:
            st.error(f"Error accessing secrets: {str(e)}")
            # Fall back to environment variables
            load_dotenv()
            api_key = os.getenv('OANDA_API_KEY')
            self.account_id = os.getenv('OANDA_ACCOUNT_ID')
            # Add 'Bearer ' prefix if not present
            self.api_key = f"Bearer {api_key}" if api_key and not api_key.startswith('Bearer ') else api_key
        
        if not self.api_key:
            st.error("""
            OANDA API key not found. Please check your Streamlit secrets or environment variables.
            
            The API key should be in this format:
            - With Bearer prefix: "Bearer 1234567890abcdef..."
            - Without Bearer prefix: "1234567890abcdef..."
            (The code will add 'Bearer ' prefix if needed)
            """)
            self.api = None
            return
            
        if not self.account_id:
            st.error("OANDA account ID not found. Please check your Streamlit secrets or environment variables.")
            self.api = None
            return
        
        try:
            # Debug: Show what we're using to initialize the API
            st.write("Initializing OANDA API with:")
            st.write("- Account ID:", self.account_id)
            st.write("- API Key length:", len(self.api_key) if self.api_key else "None")
            st.write("- API Key format:", "Bearer *" * len(api_key.replace('Bearer ', '')))
            
            # Add practice API URL for demo accounts
            self.api = oandapyV20.API(
                access_token=self.api_key.replace('Bearer ', ''),  # Remove 'Bearer ' for the API client
                environment="practice"  # Use 'practice' for demo accounts, 'live' for real accounts
            )
            self.instrument = "EUR_USD"
            
            # Test connection
            r = accounts.AccountSummary(self.account_id)
            self.api.request(r)
            st.success("Successfully connected to OANDA API")
            
        except V20Error as e:
            st.error(f"""
            Error connecting to OANDA API: {str(e)}
            
            Please check:
            1. Your API key is correct and properly formatted
            2. Your account ID is correct
            3. You're using the right environment (practice/live)
            4. Your API key has the necessary permissions
            
            Current settings:
            - Account ID: {self.account_id}
            - API Key format: {"Starts with 'Bearer'" if self.api_key and self.api_key.startswith('Bearer ') else "Missing 'Bearer' prefix"}
            
            Make sure you've added these to your Streamlit secrets:
            ```
            OANDA_API_KEY = your-api-key-here
            OANDA_ACCOUNT_ID = your-account-id-here
            ```
            
            Common issues:
            1. API key missing 'Bearer ' prefix (will be added automatically)
            2. Extra spaces or quotes in the secrets
            3. API key from wrong environment (practice vs live)
            4. Account ID from wrong environment
            """)
            self.api = None
    
    def get_account_balance(self):
        """Get current account balance"""
        try:
            if not self.api:
                st.error("OANDA API not initialized")
                return None
                
            r = accounts.AccountSummary(self.account_id)
            response = self.api.request(r)
            return float(response['account']['balance'])
            
        except V20Error as e:
            st.error(f"""
            Error getting account balance: {str(e)}
            
            Response details:
            {e.response.text if hasattr(e, 'response') and hasattr(e.response, 'text') else 'No additional details'}
            """)
            return None
    
    def get_open_positions(self):
        """Get all open positions"""
        try:
            if not self.api:
                st.error("OANDA API not initialized")
                return []
                
            r = trades.OpenTrades(self.account_id)
            response = self.api.request(r)
            positions = []
            for trade in response['trades']:
                if trade['instrument'] == self.instrument:
                    positions.append({
                        'id': trade['id'],
                        'type': 'long' if float(trade['initialUnits']) > 0 else 'short',
                        'units': abs(float(trade['initialUnits'])),
                        'entry_price': float(trade['price']),
                        'unrealized_pnl': float(trade['unrealizedPL']),
                        'entry_time': datetime.strptime(trade['openTime'].split('.')[0], '%Y-%m-%dT%H:%M:%S')
                    })
            return positions
            
        except V20Error as e:
            st.error(f"""
            Error getting open positions: {str(e)}
            
            Response details:
            {e.response.text if hasattr(e, 'response') and hasattr(e.response, 'text') else 'No additional details'}
            """)
            return []
    
    def get_trade_history(self):
        """Get trade history"""
        try:
            if not self.api:
                st.error("OANDA API not initialized")
                return []
                
            r = trades.TradesList(self.account_id)
            response = self.api.request(r)
            trades_list = []
            for trade in response['trades']:
                if trade['instrument'] == self.instrument and trade['state'] == 'CLOSED':
                    trades_list.append({
                        'id': trade['id'],
                        'type': 'long' if float(trade['initialUnits']) > 0 else 'short',
                        'units': abs(float(trade['initialUnits'])),
                        'entry_price': float(trade['price']),
                        'exit_price': float(trade['averageClosePrice']),
                        'pnl': float(trade['realizedPL']),
                        'entry_time': datetime.strptime(trade['openTime'].split('.')[0], '%Y-%m-%dT%H:%M:%S'),
                        'exit_time': datetime.strptime(trade['closeTime'].split('.')[0], '%Y-%m-%dT%H:%M:%S')
                    })
            return trades_list
            
        except V20Error as e:
            st.error(f"""
            Error getting trade history: {str(e)}
            
            Response details:
            {e.response.text if hasattr(e, 'response') and hasattr(e.response, 'text') else 'No additional details'}
            """)
            return []
    
    def place_order(self, order_type, units, current_price=None):
        """Place a market order"""
        try:
            if not self.api:
                st.error("OANDA API not initialized")
                return False
                
            order_data = {
                "order": {
                    "type": "MARKET",
                    "instrument": self.instrument,
                    "units": str(units) if order_type == 'long' else str(-units),
                    "timeInForce": "FOK",
                    "positionFill": "DEFAULT"
                }
            }
            
            r = orders.OrderCreate(self.account_id, data=order_data)
            response = self.api.request(r)
            
            if response['orderFillTransaction']['type'] == 'ORDER_FILL':
                st.success(f"Order filled: {order_type.upper()} {units} units at {response['orderFillTransaction']['price']}")
                return True
            return False
            
        except V20Error as e:
            st.error(f"""
            Error placing order: {str(e)}
            
            Response details:
            {e.response.text if hasattr(e, 'response') and hasattr(e.response, 'text') else 'No additional details'}
            
            Order details:
            Type: {order_type}
            Units: {units}
            Current price: {current_price}
            """)
            return False
    
    def close_position(self, trade_id):
        """Close a specific position"""
        try:
            if not self.api:
                st.error("OANDA API not initialized")
                return False
                
            r = trades.TradeClose(self.account_id, trade_id)
            response = self.api.request(r)
            
            if response['orderFillTransaction']['type'] == 'ORDER_FILL':
                st.success(f"Position closed at {response['orderFillTransaction']['price']}")
                return True
            return False
            
        except V20Error as e:
            st.error(f"""
            Error closing position: {str(e)}
            
            Response details:
            {e.response.text if hasattr(e, 'response') and hasattr(e.response, 'text') else 'No additional details'}
            
            Trade ID: {trade_id}
            """)
            return False
    
    def close_all_positions(self):
        """Close all open positions"""
        try:
            if not self.api:
                st.error("OANDA API not initialized")
                return False
                
            positions = self.get_open_positions()
            for position in positions:
                self.close_position(position['id'])
            return True
            
        except Exception as e:
            st.error(f"""
            Error closing all positions: {str(e)}
            
            Please check each position individually.
            """)
            return False
