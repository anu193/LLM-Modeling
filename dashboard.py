import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# ----- Page Config -----
st.set_page_config(page_title="Enhancing Business Strategies using AI - LLM", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# ----- Load Models -----
@st.cache_resource
def load_sentiment_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("./final_model")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(LABEL_MAP), pad_token_id=tokenizer.pad_token_id
        )
        model = PeftModel.from_pretrained(base_model, "./final_model").to(device).eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None, None

@st.cache_resource
def load_sales_model():
    bundle_path = "/content/drive/MyDrive/BuisinessStrat/sales_forecaster_multi.pt"
    try:
        st.warning("Loading sales model with weights_only=False. Ensure the file source is trusted to avoid security risks.")
        bundle = torch.load(bundle_path, map_location=device, weights_only=False)
        class LSTMForecaster(torch.nn.Module):
            def __init__(self, in_dim, hid_dim=32):
                super().__init__()
                self.lstm = torch.nn.LSTM(in_dim, hid_dim, batch_first=True)
                self.fc = torch.nn.Linear(hid_dim, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])
        model = LSTMForecaster(len(bundle["feature_cols"]))
        model.load_state_dict(bundle["model_state"])
        model.to(device).eval()
        stores = [str(s) for s in bundle["stores"]]
        if not stores:
            st.warning("No stores found in the model. Using default store '1' as a fallback.")
            stores = ["1"]
        return model, bundle["scalers"], stores, bundle["feature_cols"]
    except Exception as e:
        st.error(f"Error loading sales model: {e}")
        return None, None, ["1"], []

# ----- Prediction Helpers -----
def predict_sentiments(df, tokenizer, model):
    try:
        texts = df["reviewText"].fillna("").astype(str).tolist()
        results, confs = [], []
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            if not batch or all(not t.strip() for t in batch):
                continue
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            results.extend([LABEL_MAP[p] for p in preds])
            confs.extend(probs.max(axis=1).tolist())
        df_out = df.iloc[:len(results)].copy()
        df_out["sentiment"] = results
        df_out["confidence"] = confs
        return df_out
    except Exception as e:
        st.error(f"Sentiment prediction error: {e}")
        return df

def predict_sales(df, model, scalers, stores, feat_cols, selected_store, future_steps=7, window=4):
    try:
        required_cols = ["Date", "Store", "Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in sales data: {missing_cols}")

        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors='coerce')
        df = df.dropna(subset=["Date"]).sort_values(["Store", "Date"]).reset_index(drop=True)

        # Convert store IDs to strings in the DataFrame
        df["Store"] = df["Store"].astype(str)

        # Filter out unmatched stores
        valid_stores = set(stores)
        unmatched_stores = set(df["Store"]) - valid_stores
        if unmatched_stores:
            st.warning(f"Some stores in the data do not match trained stores: {unmatched_stores}. Excluding these rows. Use stores {valid_stores}.")
            df = df[df["Store"].isin(valid_stores)].reset_index(drop=True)

        # Fallback if no valid stores remain
        if df.empty and valid_stores:
            st.warning(f"No valid stores found. Using default store: {stores[0]}.")
            default_row = pd.DataFrame([[pd.to_datetime("2025-05-01"), stores[0], 0, 0, 0, 0, 0, 0]], columns=required_cols)
            default_row["Store"] = default_row["Store"].astype(str)
            df = default_row

        # Initialize Predicted_Sales column
        df["Predicted_Sales"] = np.nan

        # One-hot encode stores (only for the stores the model was trained on)
        idx_map = {s: i for i, s in enumerate(stores)}
        df["StoreIdx"] = df["Store"].map(idx_map).fillna(0).astype(int)
        oh = np.eye(len(stores))[df["StoreIdx"]]
        store_cols = [f"store_{s}" for s in stores]
        df = pd.concat([df, pd.DataFrame(oh, columns=store_cols)], axis=1)

        # Scale numeric features per store
        num_cols = ["Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
        for s in stores:
            mask = df["Store"] == s
            if mask.any() and s in scalers:
                df.loc[mask, num_cols] = scalers[s].transform(df.loc[mask, num_cols])

        # Historical sequence predictions
        seqs = []
        for i in range(len(df) - window + 1):
            seq = df.iloc[i:i+window][feat_cols].values
            if not np.isnan(seq).any():
                seqs.append(seq)
        if seqs:
            X = torch.tensor(np.stack(seqs), dtype=torch.float32).to(device)
            with torch.no_grad():
                hist = model(X).detach().cpu().numpy().flatten()  # Detach to avoid grad issues
            df.loc[window-1:window-1+len(hist)-1, 'Predicted_Sales'] = hist.tolist()

        # Future predictions
        seq = df.iloc[-window:][feat_cols].values.copy()
        future = []
        for _ in range(future_steps):
            Xf = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32).to(device)
            with torch.no_grad():
                p = model(Xf).detach().cpu().numpy().item()  # Detach to avoid grad issues
            future.append(p)
            seq = np.vstack([seq[1:], seq[-1]])
            seq[-1, feat_cols.index("Weekly_Sales")] = p
        dates = pd.date_range(start=df["Date"].max() + pd.Timedelta(1, 'D'), periods=future_steps)
        future_df = pd.DataFrame({
            "Date": dates,
            "Store": selected_store,
            "Weekly_Sales": np.nan,
            "Predicted_Sales": future
        })
        return pd.concat([df, future_df], ignore_index=True)
    except Exception as e:
        st.error(f"Sales prediction error: {e}")
        if "Predicted_Sales" not in df.columns:
            df["Predicted_Sales"] = np.nan
        return df

# ----- Dashboard UI -----
st.title("Enhancing Business Strategies using AI based Model")

# 1️⃣ Sentiment Analysis Section
st.header("🔍 Amazon Review Sentiment Analysis")
try:
    df_reviews = pd.read_csv("processed_data_large.csv")
except FileNotFoundError:
    st.error("File 'processed_data_large.csv' not found. Please upload the reviews file.")
    df_reviews = pd.DataFrame(columns=["reviewText"])
tok, s_model = load_sentiment_model()
if tok is None or s_model is None:
    st.stop()
res = predict_sentiments(df_reviews, tok, s_model)
opt = st.selectbox("Filter by Sentiment", ["All", "positive", "neutral", "negative"])
if opt != "All":
    plot_df = res[res["sentiment"] == opt]
else:
    plot_df = res
counts = plot_df["sentiment"].value_counts().reindex(["negative", "neutral", "positive"], fill_value=0)
fig1 = px.bar(
    x=counts.index,
    y=counts.values,
    labels={"x": "Sentiment", "y": "Count"},
    title="Sentiment Distribution",
    color=counts.index,
    color_discrete_map={"positive": "green", "neutral": "gray", "negative": "red"}
)
st.plotly_chart(fig1, use_container_width=True)
total_reviews = len(res)
neg_count = counts.get("negative", 0)
neu_count = counts.get("neutral", 0)
pos_count = counts.get("positive", 0)
neg_ratio = neg_count / total_reviews if total_reviews > 0 else 0
neu_ratio = neu_count / total_reviews if total_reviews > 0 else 0
pos_ratio = pos_count / total_reviews if total_reviews > 0 else 0

if opt == "All":
    st.info("This analysis provides actionable insights into Amazon customer sentiment, categorizing reviews as negative, neutral, and positive to inform strategic decision-making.")
    st.subheader("Actionable Recommendations for All Sentiments:")
    if neg_ratio > 0.3:
        st.warning("- **High Negative Sentiment (>30%)**: Conduct a root cause analysis on customer feedback to identify recurring issues. Prioritize product quality improvements and enhance customer support responsiveness.")
    elif neg_count > 0:
        st.warning("- **Negative Sentiment Present**: Review negative feedback for common themes (e.g., product defects, shipping delays). Implement targeted solutions to address these pain points and improve customer satisfaction.")
    else:
        st.success("- **No Negative Sentiment**: Excellent performance! Maintain current strategies and continue monitoring feedback to sustain this trend.")
    if neu_ratio > 0.5:
        st.info("- **High Neutral Sentiment (>50%)**: A significant portion of customers are neutral. Launch engagement campaigns, such as personalized promotions or loyalty programs, to convert neutral sentiment into positive advocacy.")
    elif neu_count > 0:
        st.info("- **Neutral Sentiment Present**: Engage neutral customers with surveys to uncover their needs and preferences. Offer incentives to encourage repeat purchases and build stronger loyalty.")
    else:
        st.success("- **No Neutral Sentiment**: Focus on maintaining positive sentiment while addressing any emerging negative feedback promptly.")
    if pos_ratio > 0.5:
        st.success("- **High Positive Sentiment (>50%)**: Strong positive feedback indicates customer satisfaction. Leverage this sentiment in marketing campaigns and explore upselling or cross-selling opportunities to maximize revenue.")
    elif pos_count > 0:
        st.success("- **Positive Sentiment Present**: Highlight positive reviews in promotional materials to build trust with prospective customers. Consider introducing referral programs to capitalize on satisfied customers.")
    else:
        st.warning("- **No Positive Sentiment**: Focus on improving customer experience through quality enhancements and better service to foster positive sentiment.")
elif opt == "positive":
    if pos_count > 0:
        st.success("Positive sentiment detected. Showcase these reviews in marketing campaigns to build trust and explore upselling opportunities to capitalize on customer satisfaction.")
    else:
        st.warning("No positive sentiment found. Focus on improving customer experience to foster positive feedback.")
elif opt == "negative":
    if neg_count > 0:
        st.warning("Negative sentiment detected. Conduct a root cause analysis to address customer pain points, such as product quality or delivery issues, and enhance support to improve satisfaction.")
    else:
        st.success("No negative sentiment found. Continue monitoring feedback to maintain this trend.")
elif opt == "neutral":
    if neu_count > 0:
        st.info("Neutral sentiment detected. Engage these customers with surveys or targeted promotions to understand their needs and convert neutrality into positive loyalty.")
    else:
        st.success("No neutral sentiment found. Focus on sustaining positive sentiment and addressing any negative feedback.")

# 2️⃣ Sales Forecasting Section
st.header("📈 Sales Forecasting & Strategy Recommendations")
try:
    df_sales = pd.read_csv("sales_data.csv")
except FileNotFoundError:
    st.error("File 'sales_data.csv' not found. Please upload the sales file.")
    df_sales = pd.DataFrame(columns=["Date", "Store", "Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"])
df_sales["Date"] = pd.to_datetime(df_sales["Date"], format="%d-%m-%Y", errors='coerce')
s_model, scalers, stores, feat_cols = load_sales_model()
if s_model is None:
    st.stop()
store = st.selectbox("Select Store", stores)
fut_date = st.date_input("Choose Forecast Date", min_value=df_sales["Date"].max() + pd.Timedelta(1, 'D'))
days = (pd.to_datetime(fut_date) - df_sales["Date"].max()).days
fc = predict_sales(df_sales, s_model, scalers, stores, feat_cols, store, future_steps=days)
pred = fc.loc[fc["Date"] == pd.to_datetime(fut_date), "Predicted_Sales"]
if pred.empty or pd.isna(pred.iloc[0]):
    st.error("Unable to generate sales forecast for the selected date.")
else:
    st.write(f"**Forecast for {store} on {fut_date}:** {pred.iloc[0]:.2f} weekly sales units")
fig2 = go.Figure()
sub = fc[fc["Store"] == store]
fig2.add_trace(go.Scatter(x=sub["Date"], y=sub["Weekly_Sales"], mode='lines+markers', name='Actual', line=dict(color='blue')))
fig2.add_trace(go.Scatter(x=sub["Date"], y=sub["Predicted_Sales"], mode='lines+markers', name='Forecast', line=dict(color='orange', dash='dash')))
fig2.update_layout(title="Historical & Forecasted Weekly Sales", xaxis_title="Date", yaxis_title="Weekly Sales", hovermode="x unified")
st.plotly_chart(fig2, use_container_width=True)
hist_avg = df_sales[df_sales["Store"] == store]["Weekly_Sales"].mean()
if not pred.empty and not pd.isna(pred.iloc[0]):
    if pred.iloc[0] > hist_avg * 1.1:
        st.success("Projected sales up >10%. Consider ramping up inventory and marketing to capture increased demand.")
    elif pred.iloc[0] < hist_avg * 0.9:
        st.warning("Projected sales down >10%. Plan promotions or adjust stock levels to mitigate potential shortfall.")
    else:
        st.info("Sales forecast within normal variance. Continue current resource allocation and monitor metrics.")
