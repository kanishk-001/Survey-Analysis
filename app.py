from pathlib import Path

import random
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"


def find_csv(path: Path):
    files = list(path.glob("*.csv"))
    return files[0] if files else None


@st.cache_data
def load_data():
    csv = find_csv(DATA_DIR)
    if csv is None:
        return None
    return pd.read_csv(csv)


@st.cache_resource
def load_models():
    models = {}
    if (MODEL_DIR / "sentiment_classifier.pkl").exists():
        models["classifier"] = joblib.load(MODEL_DIR / "sentiment_classifier.pkl")
    if (MODEL_DIR / "satisfaction_regressor.pkl").exists():
        models["regressor"] = joblib.load(MODEL_DIR / "satisfaction_regressor.pkl")
    return models


def predict_heuristics(text: str, topic: str = None):
    t = (text or '').lower()
    sentiment = 'Neutral'
    urgency = 'Low'
    if any(w in t for w in ['dangerous', 'leak', 'contaminated', 'urgent', 'smell', 'overflow', 'fire', 'accident', 'not working']):
        sentiment = 'Negative'
        urgency = 'High'
    elif any(w in t for w in ['thanks', 'resolved', 'good', 'fixed', 'improved']):
        sentiment = 'Positive'
        urgency = 'Low'
    # routing/action removed per user request
    action = 'Pending'
    if sentiment == 'Positive':
        action = 'Resolved'
    score = 3
    if sentiment == 'Negative':
        score -= 1
    if sentiment == 'Positive':
        score += 1
    if urgency == 'High':
        score -= 1
    score = max(1, min(5, score))
    return {
        'AI_Predicted_Sentiment': sentiment,
        'Urgency_Level': urgency,
        'Action_Status': action,
        'Satisfaction_Score': score,
    }


def kpis(df: pd.DataFrame):
    total = len(df)
    mean_satisfaction = None
    if 'Satisfaction_Score' in df.columns:
        mean_satisfaction = float(df['Satisfaction_Score'].mean())
    sentiment_counts = df['AI_Predicted_Sentiment'].value_counts() if 'AI_Predicted_Sentiment' in df.columns else None
    return {'total': total, 'mean_satisfaction': mean_satisfaction, 'sentiment_counts': sentiment_counts}


def page_analysis(df: pd.DataFrame):
    st.header('Analysis')
    st.markdown('Exploratory data inspection.')

    st.subheader('Data sample')
    st.dataframe(df.head())

    st.subheader('Columns')
    st.write(list(df.columns))

    st.subheader('Missing values')
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if not miss.empty:
        st.bar_chart(miss)
    else:
        st.write('No missing values detected')

    st.subheader('Column inspector')
    col = st.selectbox('Choose column to inspect', df.columns.tolist(), key='col_inspect')
    if df[col].dtype == object or df[col].dtype.name == 'category':
        vc = df[col].value_counts().reset_index()
        vc.columns = [col, 'count']
        st.dataframe(vc.head(50))
        st.bar_chart(vc.set_index(col))
    else:
        st.write(df[col].describe())
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        sns.histplot(df[col].dropna(), ax=ax[0], kde=True)
        sns.boxplot(x=df[col].dropna(), ax=ax[1])
        st.pyplot(fig)


def page_dashboards(df: pd.DataFrame):
    st.header('Dashboards')
    st.markdown('Quick high-level KPIs and visual summaries.')
    stats = kpis(df)
    c1, c2, c3 = st.columns(3)
    c1.metric('Total Responses', stats['total'])
    c2.metric('Mean Satisfaction', f"{stats['mean_satisfaction']:.2f}" if stats['mean_satisfaction'] is not None else 'N/A')
    if stats['sentiment_counts'] is not None and not stats['sentiment_counts'].empty:
        c3.metric('Top Sentiment', stats['sentiment_counts'].idxmax())

    st.subheader('Satisfaction distribution')
    if 'Satisfaction_Score' in df.columns:
        fig = plt.figure(figsize=(8, 3))
        sns.histplot(df['Satisfaction_Score'].dropna(), bins=5)
        st.pyplot(fig)

    # Additional charts
    st.subheader('Categorical breakdowns')
    cat_cols = df.select_dtypes(include=[object, 'category']).columns.tolist()
    if cat_cols:
        sel_cat = st.selectbox('Choose categorical column for breakdown', cat_cols, index=0)
        vc = df[sel_cat].value_counts().head(10)
        # small bar + pie side-by-side
        b1, b2 = st.columns(2)
        with b1:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            sns.barplot(x=vc.values, y=vc.index, ax=ax)
            ax.set_xlabel('count')
            st.pyplot(fig)
        with b2:
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.pie(vc.values, labels=vc.index, autopct='%1.1f%%')
            ax2.set_title(f'Top {sel_cat}')
            st.pyplot(fig2)
    else:
        st.write('No categorical columns found — showing value counts for first column')
        if df.shape[1] > 0:
            col0 = df.columns[0]
            vc = df[col0].astype(str).value_counts().head(10)
            fig, ax = plt.subplots(figsize=(4, 2.5))
            sns.barplot(x=vc.values, y=vc.index, ax=ax)
            st.pyplot(fig)

    st.subheader('Time series / trends')
    # detect date-like column; provide fallback trend using index or text length
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_cols:
        dc = date_cols[0]
        try:
            ts = pd.to_datetime(df[dc], errors='coerce')
            df_ts = df.copy()
            df_ts['_ts'] = ts
            df_ts = df_ts.dropna(subset=['_ts'])
            if not df_ts.empty:
                grp = df_ts.set_index('_ts').resample('M').size()
                fig, ax = plt.subplots(figsize=(6, 2.5))
                ax.plot(grp.index, grp.values, marker='o')
                ax.set_title('Responses over time (monthly)')
                st.pyplot(fig)
                # sentiment trend
                if 'AI_Predicted_Sentiment' in df.columns:
                    sentiment_ts = df_ts.groupby([pd.Grouper(key='_ts', freq='M'), 'AI_Predicted_Sentiment']).size().unstack(fill_value=0)
                    fig2, ax2 = plt.subplots(figsize=(6, 2.5))
                    sentiment_ts.plot(ax=ax2)
                    ax2.set_title('Sentiment over time')
                    st.pyplot(fig2)
        except Exception:
            st.write('Could not parse date column for time series')
    else:
        # fallback: use rolling averages over index or text-length trend
        st.write('No date/time column detected — showing fallback trend')
        if 'Satisfaction_Score' in df.columns:
            s = df['Satisfaction_Score'].dropna().reset_index(drop=True)
            if len(s) > 5:
                roll = s.rolling(window=max(3, len(s)//10)).mean()
                fig, ax = plt.subplots(figsize=(6, 2.5))
                ax.plot(roll.index, roll.values)
                ax.set_title('Rolling mean satisfaction (fallback)')
                st.pyplot(fig)
        elif 'Citizen_Feedback_Text' in df.columns:
            lens = df['Citizen_Feedback_Text'].astype(str).str.len().fillna(0)
            roll = lens.rolling(window=max(3, len(lens)//10)).mean()
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.plot(roll.index, roll.values)
            ax.set_title('Rolling mean feedback length (fallback)')
            st.pyplot(fig)
        else:
            st.write('No suitable columns for fallback trend')

    st.subheader('Sentiment by topic (stacked)')
    if 'Survey_Topic' in df.columns and 'AI_Predicted_Sentiment' in df.columns:
        cross = pd.crosstab(df['Survey_Topic'], df['AI_Predicted_Sentiment'])
        top = cross.sum(axis=1).sort_values(ascending=False).head(15).index
        cross_top = cross.loc[top]
        fig, ax = plt.subplots(figsize=(8, 4))
        cross_top.plot(kind='bar', stacked=True, ax=ax)
        ax.legend(loc='upper right')
        st.pyplot(fig)
    elif 'Survey_Topic' in df.columns and 'AI_Predicted_Sentiment' not in df.columns and 'Citizen_Feedback_Text' in df.columns:
        # create temporary sentiment from feedback text
        st.write('AI_Predicted_Sentiment not found — estimating sentiment from feedback text')
        try:
            temp_sent = df['Citizen_Feedback_Text'].astype(str).apply(lambda x: predict_heuristics(x)['AI_Predicted_Sentiment'])
            cross = pd.crosstab(df['Survey_Topic'], temp_sent)
            top = cross.sum(axis=1).sort_values(ascending=False).head(15).index
            cross_top = cross.loc[top]
            fig, ax = plt.subplots(figsize=(10, 5))
            cross_top.plot(kind='bar', stacked=True, ax=ax)
            st.pyplot(fig)
        except Exception:
            st.write('Could not compute estimated sentiment for stacked view')
    else:
        st.write('Survey_Topic not available for stacked view')

    st.subheader('Correlation heatmap')
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # fallback: create numeric features from text if needed
    if len(num_cols) < 2 and 'Citizen_Feedback_Text' in df.columns:
        df['_text_len'] = df['Citizen_Feedback_Text'].astype(str).str.len().fillna(0)
        df['_word_count'] = df['Citizen_Feedback_Text'].astype(str).str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.write('Not enough numeric columns for correlation heatmap even after fallbacks')

    st.subheader('Scatter / relationships')
    if len(num_cols) >= 2:
        x_col = st.selectbox('X axis', num_cols, index=0, key='scatter_x')
        y_col = st.selectbox('Y axis', num_cols, index=1 if len(num_cols) > 1 else 0, key='scatter_y')
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
        try:
            sns.regplot(data=df, x=x_col, y=y_col, scatter=False, ax=ax, color='red')
        except Exception:
            pass
        st.pyplot(fig)
    else:
        st.write('Not enough numeric columns for scatter plot even after fallbacks')

    st.subheader('Boxplot by category')
    if num_cols and cat_cols:
        num = st.selectbox('Numeric for boxplot', num_cols, key='box_num')
        cat = st.selectbox('Category for boxplot', cat_cols, key='box_cat')
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(x=df[cat], y=df[num], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    else:
        # fallback: bucket text length and show boxplot
        if '_text_len' in df.columns and df['_text_len'].notna().any():
            df['_len_bucket'] = pd.qcut(df['_text_len'].replace(0, np.nan).fillna(0)+1, q=4, duplicates='drop')
            if df['_len_bucket'].notna().any():
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=df['_len_bucket'].astype(str), y=df.get('Satisfaction_Score', df['_text_len']), ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                st.pyplot(fig)
            else:
                st.write('Not enough data for fallback boxplot')
        else:
            st.write('Need numeric and categorical columns for boxplot')


def page_reports(df: pd.DataFrame, models: dict):
    st.header('Reports')
    st.markdown('Select a single record from the dataset and click Predict.')

    st.subheader('Dataset summary')
    stats = kpis(df)
    st.write(f"Total responses: {stats['total']}")
    if stats['mean_satisfaction'] is not None:
        st.write(f"Mean satisfaction: {stats['mean_satisfaction']:.2f}")

    # Input selection (dataset-only)
    st.subheader('Input source')
    cols = df.columns.tolist()
    # present options as index + short preview
    choices = [f"{i}: {str(df.iloc[i][cols[0]])[:80]}" for i in range(len(df))]
    sel = st.selectbox('Choose a row', choices, index=0)
    idx = int(sel.split(':', 1)[0])
    sample = df.iloc[[idx]].copy()

    if sample is None:
        st.info('Choose or enter a record to enable prediction')
        return

    # Display input on left, actions on right
    left, right = st.columns([1, 1])
    with left:
        st.markdown('**Input record**')
        st.dataframe(sample.T)

    # run prediction
    text_field = None
    topic_field = None
    for c in sample.columns:
        if 'feedback' in c.lower() or 'text' in c.lower():
            text_field = c
        if 'topic' in c.lower():
            topic_field = c

    text = sample.iloc[0][text_field] if text_field else ''
    topic = sample.iloc[0][topic_field] if topic_field else None

    if st.button('Predict'):
        res = predict_heuristics(text, topic)
        # try model overrides
        try:
            if 'classifier' in models:
                res['AI_Predicted_Sentiment'] = str(models['classifier'].predict(sample)[0])
        except Exception:
            pass
        try:
            if 'regressor' in models:
                res['Satisfaction_Score'] = float(models['regressor'].predict(sample)[0])
        except Exception:
            pass

        # Improved UI for results
        with right:
            st.markdown('**Prediction**')
            cols_res = st.columns(3)
            cols_res[0].metric('Satisfaction', res.get('Satisfaction_Score', 'N/A'))
            sentiment = res.get('AI_Predicted_Sentiment', 'N/A')
            urgency = res.get('Urgency_Level', 'N/A')
            if sentiment == 'Positive':
                cols_res[1].success(f'Sentiment: {sentiment}')
            elif sentiment == 'Negative':
                cols_res[1].warning(f'Sentiment: {sentiment}')
            else:
                cols_res[1].info(f'Sentiment: {sentiment}')
            if urgency == 'High':
                cols_res[2].warning(f'Urgency: {urgency}')
            else:
                cols_res[2].info(f'Urgency: {urgency}')

            # Routing / Action display removed per user request

        # Recommendations removed by default; keep generate_recommendations() available in code


def generate_recommendations(row: pd.Series, prediction: dict) -> dict:
    # Very simple rule-based recommendations
    sentiment = prediction.get('AI_Predicted_Sentiment', 'Neutral')
    urgency = prediction.get('Urgency_Level', 'Low')
    score = prediction.get('Satisfaction_Score', None)

    summary = f"Detected sentiment {sentiment} with urgency {urgency}."

    # problem: derive from topic or feedback
    problem = 'General citizen issue'
    if 'Survey_Topic' in row.index and pd.notna(row['Survey_Topic']):
        problem = f"Issue related to {row['Survey_Topic']}"
    elif any('leak' in str(x).lower() or 'water' in str(x).lower() for x in row.values):
        problem = 'Water-related problem detected'

    recommendation = 'Assign to relevant department and investigate.'
    next_steps = []
    if urgency == 'High':
        recommendation = 'Immediate field inspection and escalation to senior team.'
        next_steps.append('Create high-priority ticket')
        next_steps.append('Notify relevant operations immediately')
    else:
        recommendation = 'Schedule investigation and provide citizen updates.'
        next_steps.append('Open standard service ticket')
        next_steps.append('Assign to operations team')

    if sentiment == 'Negative' and score is not None and float(score) < 3:
        next_steps.append('Prioritize case for follow-up and resolution')

    next_steps.append('Log actions and update citizen with resolution ETA')

    return {
        'summary': summary,
        'problem': problem,
        'recommendation': recommendation,
        'next_steps': next_steps,
    }


def main():
    st.set_page_config(layout='wide', page_title='Survey Analysis')
    st.title('Survey Analysis')

    df = load_data()
    models = load_models()

    if df is None:
        st.warning('No CSV found in data/. Add your dataset and reload.')
        return

    # Left sidebar navigation: click an option to render its page on the main area
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Analysis', 'Dashboards', 'Reports'], index=0)

    if page == 'Analysis':
        page_analysis(df)
    elif page == 'Dashboards':
        page_dashboards(df)
    elif page == 'Reports':
        page_reports(df, models)


if __name__ == '__main__':
    main()
