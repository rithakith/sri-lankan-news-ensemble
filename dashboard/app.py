"""Ensemble Learning for Sri Lankan News Classification — ChatGPT-assisted code."""
from pathlib import Path
import os
import json
import threading
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import (MODEL_NAMES, model_text, validate_bundle, infer, blend, read_run,
                  predictions, scores, category_scores, matrix)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get('NEWS_ARTIFACT_DIR', ROOT / 'artifacts'))
COLORS = ['#087f73', '#385ac2', '#e99b32', '#a45ab2', '#e27574', '#2c8fa8']
st.set_page_config(page_title='Sri Lankan News · Ensemble Learning', page_icon='🇱🇰', layout='wide')
st.markdown('''<style>
.block-container {padding-top:2.3rem;max-width:1450px}
h1 {letter-spacing:-.045em;font-weight:750!important}
h2,h3 {letter-spacing:-.025em}
[data-testid="stMetric"] {background:white;border:1px solid #dce4ec;border-radius:14px;padding:18px}
[data-testid="stSidebar"] {border-right:1px solid #dce4ec}
.hero-tag {font-size:14px;color:#087f73;font-weight:700;letter-spacing:.12em;text-transform:uppercase}
.hero-copy {font-size:18px;color:#50647a;max-width:780px;line-height:1.6}
</style>''', unsafe_allow_html=True)


def chart(fig):
    fig.update_layout(font=dict(family='sans-serif', size=14, color='#14283f'),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=15, r=15, t=35, b=15), legend_title_text='')
    st.plotly_chart(fig, use_container_width=True)


@st.cache_resource
def load_models(path, modified):
    import joblib
    return validate_bundle(joblib.load(path)), threading.RLock()


def get_models():
    path = ARTIFACTS / 'hybrid_bundle.joblib'
    if not path.exists():
        return None
    return load_models(str(path), path.stat().st_mtime_ns)


def get_frames(run):
    return {split: predictions(ARTIFACTS, split, run['categories']) for split in ['validation', 'test']}


def csv_download(frame, name):
    st.download_button('Download table as CSV', frame.to_csv(index=False).encode(), name,
                       mime='text/csv', key=name)


try:
    run = read_run(ARTIFACTS)
    frames = get_frames(run) if run else {}
    if run:
        for split, frame in frames.items():
            expected = run.get('comparison_counts', {}).get(split)
            if expected is not None and len(frame) != expected:
                raise ValueError(f'{split} count differs from run metadata.')
        if not frames['test'].empty and not frames['validation'].empty:
            if set(frames['test'].doc_id) & set(frames['validation'].doc_id):
                raise ValueError('Validation and test articles overlap.')
except (ValueError, KeyError, OSError) as exc:
    st.error(f'Cannot read experiment: {exc}')
    st.stop()

with st.sidebar:
    st.markdown('### 🇱🇰 News classification')
    st.caption('ENSEMBLE LEARNING LAB')
    page = st.radio('Workspace', ['Overview', 'Classify an article', 'Compare models', 'Explore errors', 'Hybrid lab', 'Methodology'], label_visibility='collapsed')
    st.divider()
    if run:
        st.markdown(f"**Run:** {run['mode']}")
        st.caption(f"{len(run['categories'])} model categories · {run.get('taxonomy_version', 'unspecified')} taxonomy")
        st.caption(f"Trained model: {'available' if (ARTIFACTS / 'hybrid_bundle.joblib').exists() else 'not installed'}")
    else:
        st.caption('Waiting for a completed experiment export')
    st.caption('R. K. Edirisinghe · 224048G')

st.markdown('<div class="hero-tag">Sri Lanka · Applied machine learning</div>', unsafe_allow_html=True)
st.title({'Overview':'One story. Six model perspectives.', 'Classify an article':'Put the models to work.',
          'Compare models':'Compare the evidence.', 'Explore errors':'Look beyond the headline score.',
          'Hybrid lab':'When does combining models help?', 'Methodology':'From local news to model decisions.'}[page])
if run:
    if str(run['mode']).startswith('pilot'):
        st.warning('PILOT EXPERIMENT · Small development sample with BART-generated reference labels. These results do not establish real-world topic accuracy.')
    else:
        st.info('Evaluation uses BART-generated reference labels. Agreement measures consistency with those labels; it is not independently verified topic accuracy.')

if page == 'Overview':
    st.markdown('<p class="hero-copy">Explore how bagging, boosting, stacking and a neural hybrid classify Sri Lankan news—and where their decisions differ.</p>', unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    a.metric('Approaches compared', '6')
    b.metric('Model categories', len(run['categories']) if run else 'Pending')
    c.metric('Training articles', f"{run['comparison_counts']['train']:,}" if run else 'Pending')
    d.metric('Test articles', len(frames['test']) if run else 'Pending')
    st.subheader('The experiment at a glance')
    left, right = st.columns([1.5,1])
    with left:
        st.markdown('''**Bagging · Random Forest** — pools bootstrapped trees with random feature subsets.

**Boosting · XGBoost** — adds trees sequentially to reduce the training objective.

**Stacking · LR + RF + XGBoost** — an LR meta-model learns from out-of-fold probability predictions.

**Hybrid · Stacking + ANN** — combines two probability distributions with a validation-selected weight.''')
    with right:
        st.markdown('**Start exploring**')
        st.write('Paste an article, compare model scores, then inspect the disagreements behind them.')
        st.caption('A Logistic Regression baseline and a separate ANN provide context for the ensemble results.')
    if run and not frames['test'].empty:
        table = scores(frames['test'], run['categories'])
        chart(px.bar(table, x='Model', y='Macro-F1', color='Model', color_discrete_sequence=COLORS,
                     range_y=[0,1], title='Frozen-model test comparison'))
        st.caption(f"Same {len(frames['test'])} test articles and {len(run['categories'])} categories for every model. Macro-F1 includes unsupported experiment categories as zero, following the notebook.")
    elif not run:
        st.info('No experiment export has been installed yet. Live prediction and numerical comparisons become available when the trained bundle and matching results are added.')
        st.code('python scripts/import_dashboard_export.py /path/to/dashboard_export.zip')

elif page == 'Classify an article':
    st.write('Enter English-language Sri Lankan news. The models use the headline and first 200 body words, matching training.')
    with st.form('article'):
        headline = st.text_input('Headline', max_chars=1000, placeholder='Enter the news headline')
        body = st.text_area('Article body', height=180, max_chars=50000, placeholder='Paste the article text here…')
        submitted = st.form_submit_button('Compare predictions', type='primary')
    if submitted:
        if not headline.strip() or not body.strip():
            st.warning('Enter both a headline and an article body.')
        else:
            try:
                loaded = get_models()
                if loaded is None or run is None:
                    st.info('The trained model export must be installed before live classification is available.')
                else:
                    bundle, lock = loaded
                    if list(bundle['label_encoder'].classes_) != run['categories']:
                        raise ValueError('Model categories differ from the experiment metadata.')
                    with lock:
                        probabilities, timing = infer(bundle, [model_text(headline, body)])
                    st.session_state['prediction'] = {'probs': probabilities, 'timing': timing, 'headline': headline}
            except Exception as exc:
                st.error(f'Prediction unavailable: {exc}')
    result = st.session_state.get('prediction')
    if result and run:
        st.subheader(result['headline'])
        categories = run['categories']
        rows = [{'Model': name, 'Predicted topic': categories[int(result['probs'][name][0].argmax())],
                 'Top model probability': float(result['probs'][name][0].max()),
                 'Inference ms': result['timing'][name]} for name in MODEL_NAMES]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={'Top model probability': st.column_config.NumberColumn(format='%.3f'),
                                    'Inference ms': st.column_config.NumberColumn(format='%.2f')})
        selected = st.selectbox('Inspect category probabilities', MODEL_NAMES, index=2)
        chart(px.bar(pd.DataFrame({'Category':categories,'Probability':result['probs'][selected][0]}),
                     x='Probability',y='Category',orientation='h',range_x=[0,1],color_discrete_sequence=COLORS))
        st.caption('Probabilities are model scores, not calibrated certainty. Timings exclude model loading; stack and hybrid include component inference costs using reused predictions.')
        csv_download(pd.DataFrame(rows), 'article_predictions.csv')

elif page in ['Compare models', 'Explore errors']:
    if not run or all(f.empty for f in frames.values()):
        st.info('Install the matching article-level evaluation export to explore measured results.')
    else:
        split = st.radio('Evaluation split', [s for s,f in frames.items() if not f.empty], horizontal=True,
                         index=1 if all(not f.empty for f in frames.values()) else 0)
        frame = frames[split]
        st.caption(f"{len(frame)} articles · {len(run['categories'])} experiment categories · same articles for every model")
        if split == 'validation':
            st.caption('Development results: ANN epoch and hybrid weight were selected on validation.')
        if page == 'Compare models':
            table = scores(frame, run['categories'])
            metric = st.selectbox('Comparison metric', ['Macro-F1','Weighted-F1','Agreement'])
            chart(px.bar(table.sort_values(metric,ascending=False),x='Model',y=metric,color='Model',
                         range_y=[0,1],color_discrete_sequence=COLORS))
            st.dataframe(table,hide_index=True,use_container_width=True)
            best = table.loc[table[metric].idxmax()]
            st.write(f"Highest observed {metric}: **{best['Model']} ({best[metric]:.3f})**.")
            st.caption(f'One changed prediction moves agreement by {1/len(frame):.1%}. A ranking on this sample does not establish superiority.')
            csv_download(table, f'{split}_metrics.csv')
            if all(not f.empty for f in frames.values()):
                stages = pd.concat([scores(f,run['categories']).assign(Split=s) for s,f in frames.items()])
                chart(px.bar(stages,x='Model',y='Macro-F1',color='Split',barmode='group',range_y=[0,1],
                             title='Validation versus frozen-model test',color_discrete_sequence=COLORS))
        else:
            model = st.selectbox('Model', MODEL_NAMES)
            cats = run['categories']
            cm = matrix(frame,model,cats)
            fig = go.Figure(go.Heatmap(z=cm,x=cats,y=cats,colorscale='Teal',zmin=0,
                                      zmax=int(frame.BART_reference.value_counts().max()),
                                      text=cm,texttemplate='%{text}',hovertemplate='Reference: %{y}<br>Predicted: %{x}<br>Articles: %{z}<extra></extra>'))
            fig.update_layout(xaxis_title='Predicted topic',yaxis_title='BART reference topic',height=550,yaxis_autorange='reversed')
            chart(fig)
            details = category_scores(frame,model,cats)
            st.dataframe(details, hide_index=True,use_container_width=True)
            st.caption('Blank category F1 means no reference examples. Overall macro-F1 follows the notebook’s all-category convention.')
            disagreement = np.array([[np.mean(frame[a]!=frame[b]) for b in MODEL_NAMES] for a in MODEL_NAMES])
            chart(go.Figure(go.Heatmap(z=disagreement,x=MODEL_NAMES,y=MODEL_NAMES,zmin=0,zmax=1,
                                      colorscale='Oranges',text=disagreement,texttemplate='%{text:.0%}')))
            st.caption('Disagreement measures different predictions, not useful diversity or correctness.')
            only = st.checkbox('Show only articles where models disagree')
            category = st.selectbox('Reference category', ['All categories',*cats])
            query = st.text_input('Search headline')
            shown = frame.copy()
            if only: shown = shown[shown[MODEL_NAMES].nunique(axis=1)>1]
            if category != 'All categories': shown = shown[shown.BART_reference==category]
            if query: shown = shown[shown.headline.str.contains(query,case=False,regex=False)]
            st.dataframe(shown,hide_index=True,use_container_width=True)
            csv_download(shown,f'{split}_article_inspection.csv')

elif page == 'Hybrid lab':
    st.write('The saved hybrid combines stacking and ANN probabilities. Validation selects the weight; frozen test reports keep that choice unchanged.')
    if not run:
        st.info('Hybrid analysis becomes available when an experiment export is installed.')
    else:
        weight = float(run['stack_weight'])
        a,b = st.columns(2)
        a.metric('Saved stacking weight',f'{weight:.0%}')
        b.metric('Saved ANN weight',f'{1-weight:.0%}')
        if weight in (0,1): st.info('Validation selected a single-model endpoint. This run does not demonstrate a hybrid benefit.')
        path = ARTIFACTS / 'blend_validation_search.csv'
        if path.exists():
            search = pd.read_csv(path)
            fig = px.line(search,x='stack_weight',y=['macro_f1','validation_agreement'],markers=True,
                          range_y=[0,1],color_discrete_sequence=COLORS,title='Validation weight search')
            fig.add_vline(x=weight,line_dash='dash')
            chart(fig)
        history_path = ARTIFACTS / 'ann_training_history.csv'
        if history_path.exists():
            hist = pd.read_csv(history_path)
            chart(px.line(hist,x='epoch',y=['training_loss','validation_log_loss'],
                          color_discrete_sequence=COLORS,title='ANN learning history'))
            st.caption('Training objective includes regularisation. Validation log loss selects the epoch; these curves do not prove generalisation.')
        st.subheader('Explore a new article’s blend')
        result = st.session_state.get('prediction')
        if not result:
            st.info('Classify an article first, then return here to explore its predictions.')
        else:
            st.write(result['headline'])
            exploratory = st.slider('Exploratory stacking weight',0.0,1.0,weight,0.05)
            probs = blend(result['probs']['Stacking'],result['probs']['ANN'],exploratory)[0]
            chart(px.bar(pd.DataFrame({'Category':run['categories'],'Probability':probs}),x='Category',
                         y='Probability',range_y=[0,1],color_discrete_sequence=COLORS))
            st.caption('Exploration only: this slider does not alter saved models, selected weights, or reported evaluation scores.')

elif page == 'Methodology':
    st.subheader('A local application: organising Sri Lankan news')
    st.write('The source corpus was scraped by Nuwan through the lk_news project. This experiment uses English-language Sri Lankan articles and predicts one primary topic per article.')
    st.link_button('View original data project','https://github.com/nuuuwan/lk_news/tree/data')
    st.graphviz_chart('''digraph {rankdir=LR;node [shape=box style="rounded,filled" fillcolor="#edf6f4" color="#087f73" fontname="sans-serif"]; "Sri Lankan news" -> "Quality screening" -> "BART reference labels" -> "Grouped splits"; "Grouped splits" -> "Training only TF-IDF"; "Training only TF-IDF" -> "LR / RF / XGBoost"; "LR / RF / XGBoost" -> "OOF probabilities" -> "LR meta-model"; "Grouped splits" -> "TF-IDF / SVD / scaling" -> "ANN"; "LR meta-model" -> "Weighted hybrid"; "ANN" -> "Weighted hybrid";}''')
    st.markdown('''**Reference labels.** BART zero-shot classification supplies labels needed for supervised training. It is the reference labelling model, not an independent ground-truth benchmark.

**Leakage controls.** Normalised exact headline or body groups stay within a split. TF-IDF is refitted inside each stacking fold. ANN preprocessing is fitted on training data. Exact grouping does not remove every related-story overlap.

**Evaluation.** Models are compared on the same eligible articles and category set. Validation selects the ANN epoch and hybrid weight; the test evaluation uses frozen settings. A small pilot previously used during taxonomy development remains a workflow check.

**Scope.** English text, one primary category, headline plus 200 body words. Topic overlap is possible. The application does not perform fact checking or sentiment analysis.

**AI assistance.** Notebook and application code were generated and revised with assistance from ChatGPT. The submitted analysis should be reviewed and understood by the author.''')
    if run:
        st.subheader('Experiment record')
        st.json(run)
        support_path = ARTIFACTS / 'category_support.csv'
        if support_path.exists():
            support = pd.read_csv(support_path)
            chart(px.bar(support,x='Category',y=['train','validation','test'],barmode='group',
                         color_discrete_sequence=COLORS,title='Category support in this comparison'))

st.divider()
st.caption('Ensemble Learning for Sri Lankan News Classification · Bagging / Boosting / Stacking / Hybridization')
