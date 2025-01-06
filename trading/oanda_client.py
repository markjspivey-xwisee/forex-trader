import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.exceptions import V20Error
import streamlit as st
import os
from datetime import datetime

class OandaClient:
    def __init__(self):
        self.api_key = os.getenv('OANDA_API_KEY')
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        self.api = oandapyV20.API(access_token=self.api_key)
        self.instrument = "EUR_USD"
    
    def get_account_balance(self):
        """Get current account balance"""
        try:
            r = accounts.AccountSummary(self.account_id)
            response = self.api.request(r)
            return float(response['account']['balance'])
        except V20Error as e:
            st.error(f"Error getting account balance: {str(e)}")
            return None
    
    def get_open_positions(self):
        """Get all open positions"""
        try:
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
            st.error(f"Error getting open positions: {str(e)}")
            return []
    
    def get_trade_history(self):
        """Get trade history"""
        try:
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
            st.error(f"Error getting trade history: {str(e)}")
            return []
    
    def place_order(self, order_type, units, current_price=None):
        """Place a market order"""
        try:
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
            st.error(f"Error placing order: {str(e)}")
            return False
    
    def close_position(self, trade_id):
        """Close a specific position"""
        try:
            r = trades.TradeClose(self.account_id, trade_id)
            response = self.api.request(r)
            
            if response['orderFillTransaction']['type'] == 'ORDER_FILL':
                st.success(f"Position closed at {response['orderFillTransaction']['price']}")
                return True
            return False
            
        except V20Error as e:
            st.error(f"Error closing position: {str(e)}")
            return False
    
    def close_all_positions(self):
        """Close all open positions"""
        try:
            positions = self.get_open_positions()
            for position in positions:
                self.close_position(position['id'])
            return True
        except Exception as e:
            st.error(f"Error closing all positions: {str(e)}")
            return False
