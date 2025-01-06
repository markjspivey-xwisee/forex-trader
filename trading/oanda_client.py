import v20
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
            self.api_key = st.secrets["OANDA_API_KEY"]  # Direct dictionary access
            self.account_id = st.secrets["OANDA_ACCOUNT_ID"]  # Direct dictionary access
            
            # Debug: Show what we got
            st.write("Raw API Key length:", len(self.api_key) if self.api_key else "Not found")
            st.write("Raw Account ID:", self.account_id if self.account_id else "Not found")
                
        except Exception as e:
            st.error(f"Error accessing secrets: {str(e)}")
            # Fall back to environment variables
            load_dotenv()
            self.api_key = os.getenv('OANDA_API_KEY')
            self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        
        if not self.api_key:
            st.error("""
            OANDA API key not found. Please check your Streamlit secrets or environment variables.
            
            Add your credentials in TOML format:
            ```toml
            [general]
            OANDA_API_KEY = "your-api-key-here"
            OANDA_ACCOUNT_ID = "your-account-id-here"
            ```
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
            st.write("- API Key length:", len(self.api_key))
            
            # Initialize API client
            self.api = v20.Context(
                hostname='api-fxpractice.oanda.com',  # Use 'api-fxtrade.oanda.com' for live
                token=self.api_key,
                port=443
            )
            self.instrument = "EUR_USD"
            
            # Test connection
            response = self.api.account.summary(self.account_id)
            if response.status != 200:
                raise Exception(f"Status {response.status}: {response.body}")
                
            st.success("Successfully connected to OANDA API")
            
        except Exception as e:
            st.error(f"""
            Error connecting to OANDA API: {str(e)}
            
            Please check:
            1. Your API key is correct and properly formatted
            2. Your account ID is correct
            3. You're using the right environment (practice/live)
            4. Your API key has the necessary permissions
            
            Current settings:
            - Account ID: {self.account_id}
            
            Make sure you've added these to your Streamlit secrets in TOML format:
            ```toml
            [general]
            OANDA_API_KEY = "your-api-key-here"
            OANDA_ACCOUNT_ID = "your-account-id-here"
            ```
            """)
            self.api = None
    
    def get_account_balance(self):
        """Get current account balance"""
        try:
            if not self.api:
                st.error("OANDA API not initialized")
                return None
                
            response = self.api.account.summary(self.account_id)
            if response.status != 200:
                raise Exception(f"Status {response.status}: {response.body}")
                
            return float(response.body['account']['balance'])
            
        except Exception as e:
            st.error(f"Error getting account balance: {str(e)}")
            return None
    
    def get_open_positions(self):
        """Get all open positions"""
        try:
            if not self.api:
                st.error("OANDA API not initialized")
                return []
                
            response = self.api.trade.list_open(self.account_id)
            if response.status != 200:
                raise Exception(f"Status {response.status}: {response.body}")
                
            positions = []
            for trade in response.body['trades']:
                if trade.instrument == self.instrument:
                    positions.append({
                        'id': trade.id,
                        'type': 'long' if float(trade.initialUnits) > 0 else 'short',
                        'units': abs(float(trade.initialUnits)),
                        'entry_price': float(trade.price),
                        'unrealized_pnl': float(trade.unrealizedPL),
                        'entry_time': datetime.strptime(trade.openTime.split('.')[0], '%Y-%m-%dT%H:%M:%S')
                    })
            return positions
            
        except Exception as e:
            st.error(f"Error getting open positions: {str(e)}")
            return []
    
    def get_trade_history(self):
        """Get trade history"""
        try:
            if not self.api:
                st.error("OANDA API not initialized")
                return []
                
            response = self.api.trade.list(self.account_id)
            if response.status != 200:
                raise Exception(f"Status {response.status}: {response.body}")
                
            trades_list = []
            for trade in response.body['trades']:
                if trade.instrument == self.instrument and trade.state == 'CLOSED':
                    trades_list.append({
                        'id': trade.id,
                        'type': 'long' if float(trade.initialUnits) > 0 else 'short',
                        'units': abs(float(trade.initialUnits)),
                        'entry_price': float(trade.price),
                        'exit_price': float(trade.averageClosePrice),
                        'pnl': float(trade.realizedPL),
                        'entry_time': datetime.strptime(trade.openTime.split('.')[0], '%Y-%m-%dT%H:%M:%S'),
                        'exit_time': datetime.strptime(trade.closeTime.split('.')[0], '%Y-%m-%dT%H:%M:%S')
                    })
            return trades_list
            
        except Exception as e:
            st.error(f"Error getting trade history: {str(e)}")
            return []
    
    def place_order(self, order_type, units, current_price=None):
        """Place a market order"""
        try:
            if not self.api:
                st.error("OANDA API not initialized")
                return False
                
            order = {
                "type": "MARKET",
                "instrument": self.instrument,
                "units": str(units) if order_type == 'long' else str(-units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT"
            }
            
            response = self.api.order.create(self.account_id, order)
            if response.status != 201:
                raise Exception(f"Status {response.status}: {response.body}")
                
            if hasattr(response.body, 'orderFillTransaction'):
                st.success(f"Order filled: {order_type.upper()} {units} units at {response.body.orderFillTransaction.price}")
                return True
            return False
            
        except Exception as e:
            st.error(f"""
            Error placing order: {str(e)}
            
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
                
            response = self.api.trade.close(self.account_id, trade_id)
            if response.status != 200:
                raise Exception(f"Status {response.status}: {response.body}")
                
            if hasattr(response.body, 'orderFillTransaction'):
                st.success(f"Position closed at {response.body.orderFillTransaction.price}")
                return True
            return False
            
        except Exception as e:
            st.error(f"""
            Error closing position: {str(e)}
            
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
