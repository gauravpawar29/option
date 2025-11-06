"""
Streamlit app: Illiquid Option / Algo Simulation (Auto-run + Timer)
Run with:
    pip install streamlit pandas plotly
    streamlit run streamlit_predatory_sim.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO
import time

st.set_page_config(page_title="Illiquid Option + Algo Simulation", layout="wide")

st.title("Illiquid Option Market — Algo vs Human Simulation (Educational)")
st.markdown(
    """
    This demo simulates a simplified illiquid options market where an algorithm
    moves the price up, then sells back to a human buyer at inflated prices.

    ⚠ Educational only. Market manipulation in real life is illegal.
    """
)

# Sidebar controls
st.sidebar.header("Market / Simulation Parameters")
initial_bid = st.sidebar.number_input("Initial bid", value=20.0, step=1.0, format="%.2f")
initial_ask = st.sidebar.number_input("Initial ask", value=100.0, step=1.0, format="%.2f")
fair_price = st.sidebar.number_input("Fair price", value=40.0, step=1.0, format="%.2f")
threshold_pct = st.sidebar.slider("Algo flip threshold (% above fair)", 0, 200, 20) / 100.0
human_buy_intent = st.sidebar.number_input("Human intended buy price", value=21.0, format="%.2f")
max_steps = st.sidebar.slider("Max simulation steps", 10, 500, 60)

st.sidebar.markdown("---")
st.sidebar.header("Algo Behavior Adjustments")
algo_aggression = st.sidebar.slider("Algo buy aggressiveness (0=gentle, 1=aggressive)", 0.0, 1.0, 0.35)
algo_trade_size = st.sidebar.number_input("Algo trade size (units per action)", value=1.0, step=1.0)
human_max_units = st.sidebar.number_input("Human max units to buy", value=3.0, step=1.0)

# Placeholder for live table and graph
table_placeholder = st.empty()
graph_placeholder = st.empty()


def run_simulation_live(initial_bid, initial_ask, fair_price, threshold_pct, human_buy_intent,
                        max_steps, algo_aggression, algo_trade_size, human_max_units, delay=0.2):

    events = []
    bid = float(initial_bid)
    ask = float(initial_ask)
    threshold = fair_price * (1 + threshold_pct)
    last_price = None

    def record(step, actor, action, price, bid, ask, note=""):
        events.append({
            "step": step,
            "actor": actor,
            "action": action,
            "price": round(price, 2) if price is not None else None,
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "note": note,
        })
        df = pd.DataFrame(events)
        table_placeholder.dataframe(df)
        df_plot = df.copy()
        df_plot["price_filled"] = df_plot["price"].fillna(method='ffill')
        fig = px.line(df_plot, x="step", y="price_filled", title="Simulated Trade Prices Over Time")
        fig.update_layout(
            yaxis_title="Price",
            xaxis_title="Step",
            legend_title="Legend",
        )
        fig.add_hline(y=fair_price, line_dash="dash", line_color="green", annotation_text="Fair price")
        fig.add_hline(y=fair_price * (1 + threshold_pct), line_dash="dot", line_color="red", annotation_text="Threshold")
        graph_placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(delay)

    record(0, "market", "initial_quotes", None, bid, ask, "illiquid wide spread")
    record(1, "human", "place_buy_intent", human_buy_intent, bid, ask, "human willing to buy")

    step = 2
    algo_side = "buy"
    human_position = 0.0
    algo_position = 0.0

    while step < max_steps:
        if algo_side == "buy":
            delta = max(0.2, (human_buy_intent - bid) * algo_aggression)
            new_bid = bid + delta
            trade_price = new_bid
            last_price = trade_price
            bid = new_bid
            ask = max(ask, last_price + 15)
            algo_position += algo_trade_size
            record(step, "algo", "buy_to_push", trade_price, bid, ask, "algo buys to create momentum")
            step += 1

            if trade_price <= human_buy_intent + 30 and human_position < human_max_units:
                human_buy_price = max(human_buy_intent, trade_price)
                human_position += 1.0
                record(step, "human", "buy_filled", human_buy_price, bid, ask, "human receives partial fill")
                step += 1

            if last_price >= threshold:
                algo_side = "sell"
                record(step, "algo", "flip_to_sell", None, bid, ask, f"price reached threshold {threshold:.2f}")
                step += 1

        else:
            sell_price = max(ask - 3, (last_price or ask) * 1.02)
            last_price = sell_price
            ask = sell_price
            algo_position -= algo_trade_size
            record(step, "algo", "sell_to_human_or_book", sell_price, bid, ask, "algo sells into demand")
            step += 1

            if human_position < human_max_units:
                human_position += 1.0
                record(step, "human", "buy_filled_at_loss", sell_price, bid, ask,
                       "human buys at inflated price (loss vs fair)")
                step += 1

            if algo_position <= 0:
                bid = float(initial_bid)
                ask = float(initial_ask)
                record(step, "algo", "reset_quotes", None, bid, ask, "algo returns to wide quotes")
                step += 1
                break

    record(step, "summary", "positions", None, bid, ask,
           f"human_position={human_position}, algo_position={algo_position}, last_price={last_price}")

    return pd.DataFrame(events)


# Run the simulation automatically on page load
df = run_simulation_live(initial_bid, initial_ask, fair_price, threshold_pct,
                         human_buy_intent, max_steps, algo_aggression, algo_trade_size, human_max_units)

# CSV download
csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False)
st.download_button("Download events CSV", data=csv_buffer.getvalue().encode("utf-8"),
                   file_name="simulation_events.csv")

st.caption("Educational only — do not use for real market manipulation.")
