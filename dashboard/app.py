"""Compact dashboard backed entirely by the completed experiment."""
import pandas as pd
import plotly.express as px
import streamlit as st
from core import AUDIT_NAMES, STAGES, audit_matrix, load_classifier, predict_article, selection, table

st.set_page_config(page_title='NewsLab · Sri Lankan News', page_icon='📰', layout='wide')
st.markdown('''<style>.block-container{max-width:1200px;padding-top:2.5rem}
h1{letter-spacing:-.04em}[data-testid=stMetric]{border:1px solid #dde6eb;padding:1rem;border-radius:12px}</style>''', unsafe_allow_html=True)
st.sidebar.title('NewsLab')
st.sidebar.caption('SRI LANKAN NEWS CLASSIFICATION')
page = st.sidebar.radio('Explore', ['Results', 'Predict an article', 'About the study'])
st.sidebar.divider()
st.sidebar.caption('TF-IDF + LinearSVC · selected refined\n\n13 news categories')
frozen = selection()
topics = frozen['categories']

def pretty(value):
    return value.replace('_', ' ').title()

@st.cache_resource
def classifier():
    return load_classifier()

if page == 'Results':
    st.title('Less manual labeling. Measured results.')
    st.caption('One completed experiment on 10,000 Sri Lankan news articles. No simulated scores.')
    audit = table('human_audit_metrics.csv').query("weighting == 'unweighted audit'")
    final = audit.loc[audit.model.eq('Final classifier')].iloc[0]
    bart = audit.loc[audit.model.eq('BART')].iloc[0]
    test = table('final_test_metrics.csv').iloc[0]
    cols = st.columns(3)
    cols[0].metric('Reviewed-label accuracy', f'{final.accuracy:.0%}', f'{100*(final.accuracy-bart.accuracy):.0f} pp vs BART')
    cols[1].metric('Reviewed-label macro F1', f'{final.macro_f1:.3f}')
    cols[2].metric('Test agreement with BART', f'{test.accuracy:.1%}')
    st.caption('192/300 reviewed labels matched. BART agreement uses 1,941 different test articles; it is not human-verified accuracy.')
    st.subheader('Which models performed best?')
    view = st.radio('Evaluation', ['Reviewed audit', 'Validation experiments'], horizontal=True)
    if view == 'Reviewed audit':
        shown = audit.assign(Model=audit.model.map(AUDIT_NAMES))
        long = shown.melt(id_vars='Model', value_vars=['accuracy', 'macro_f1'], var_name='Measure', value_name='Score')
        fig = px.bar(long, x='Model', y='Score', color='Measure', barmode='group',
                     color_discrete_sequence=['#397da8', '#e69b45'], range_y=[0, 1])
        fig.update_layout(legend_title_text='', xaxis_title='', margin=dict(t=15, b=10))
        st.plotly_chart(fig, width='stretch')
        st.caption('Same 300 reviewed labels for every model. LinearSVC has higher accuracy; BART has slightly higher macro F1. The audit did not select the model.')
    else:
        compared = table('model_comparison_validation.csv')
        compared['Stage'] = compared.stage.map(STAGES)
        fig = px.bar(compared, x='model', y='macro_f1', color='Stage', barmode='group',
                     category_orders={'Stage': list(STAGES.values())}, range_y=[0, 1],
                     color_discrete_sequence=['#b7d4e8', '#78add0', '#397da8', '#154b73'])
        fig.update_layout(yaxis_title='Validation macro F1', xaxis_title='', margin=dict(t=15, b=10))
        st.plotly_chart(fig, width='stretch')
        st.caption('Five classifiers × four stages, evaluated against BART pseudo-labels. Selected-refined LinearSVC won at macro F1 0.642. Stacking and weighting did not beat it.')
    st.subheader('Where do predictions disagree?')
    choice = st.selectbox('Heatmap comparison', ['Human labels vs LinearSVC', 'Human labels vs BART', 'LinearSVC vs BART'])
    refs = {'Human labels vs LinearSVC': ('human_label', 'final_prediction'),
            'Human labels vs BART': ('human_label', 'bart_label'),
            'LinearSVC vs BART': ('bart_label', 'final_prediction')}
    reference, prediction = refs[choice]
    labels = [pretty(t) for t in topics]
    fig = px.imshow(audit_matrix(reference, prediction, topics), x=labels, y=labels,
                    color_continuous_scale='Blues', zmin=0, zmax=1, text_auto='.2f', aspect='auto')
    fig.update_layout(height=650, xaxis_title='Predicted category',
                      yaxis_title='BART reference' if reference == 'bart_label' else 'Human-reviewed category',
                      margin=dict(t=10, b=10), coloraxis_colorbar_title='Row fraction')
    st.plotly_chart(fig, width='stretch')
    st.caption('Rows sum to one. Diagonal cells show category recall against the chosen reference. SVC vs BART measures agreement, not correctness.')
    with st.expander('Key findings and uncertainty'):
        st.write('Refinement added 315 articles to 3,156 seeds (about 10%). Four of five classifiers improved validation macro F1; XGBoost declined.')
        ci = table('human_audit_bootstrap_intervals.csv').query("model == 'Final classifier' and weighting == 'unweighted audit'").iloc[0]
        st.write(f'Descriptive 95% bootstrap interval for reviewed-label accuracy: {ci.accuracy_low:.1%}–{ci.accuracy_high:.1%}. Small category samples and one supplied review set limit certainty.')
        st.write('Refined stacking corrected 33 BART errors and introduced 30: a net gain of three. These counts describe stacking, not LinearSVC.')
    st.download_button('Download model comparison', table('model_comparison_validation.csv').to_csv(index=False), 'model_comparison_validation.csv', 'text/csv')

elif page == 'Predict an article':
    st.title('Try the trained classifier')
    st.caption('Frozen TF-IDF + selected-refined LinearSVC. No retraining or external API call.')
    with st.form('article'):
        headline = st.text_input('Headline', max_chars=1000)
        body = st.text_area('Article body', height=220, max_chars=50000)
        submitted = st.form_submit_button('Predict category', type='primary')
    if submitted:
        if not (headline.strip() or body.strip()):
            st.warning('Enter a headline or article body first.')
        else:
            try:
                result = predict_article(classifier(), headline, body)
                st.subheader(pretty(result['category']))
                if result['known_terms'] == 0:
                    st.warning('No terms matched the training vocabulary. Try an English news article.')
                elif result['margin'] < 0.2:
                    st.info('The top two scores are close. Review this prediction manually.')
                st.dataframe(result['alternatives'].assign(Category=lambda f: f.Category.map(pretty)), hide_index=True, width='stretch')
                st.caption(f"Top-two decision margin: {result['margin']:.3f}. Decision scores are not accuracy percentages or calibrated probabilities. The review cutoff is a heuristic.")
            except (ValueError, OSError, KeyError) as exc:
                st.error(f'Unable to use the saved classifier: {exc}')

else:
    st.title('About the study')
    st.write('BART supplies initial labels. Confident seeds train logistic regression, random forest, XGBoost, LinearSVC and stacking. A cautious refinement rule adds accepted pseudo-labels from the uncertain training pool.')
    st.markdown('**Flow:** group duplicates → reserve review groups → split → train → refine → select on validation → freeze → evaluate.')
    st.subheader('Four training stages')
    st.dataframe(pd.DataFrame({'Stage': list(STAGES.values()), 'Meaning': [
        'Confident BART seeds only', 'Original confidence/voting rule adds articles',
        'Validation-screened refinement rule', 'Added pseudo-labels receive lower weight']}), hide_index=True, width='stretch')
    st.subheader('What the results support')
    st.write('The pipeline reduces manual training annotation and shows modest, model-dependent gains. Shared BART supervision, weak minority categories, a small review set and previous development exposure limit conclusions.')
    st.caption('The completed audit was identified by the user as reviewed. Original annotation notes and AI-assisted provenance are retained in the audit export; no second independent annotator was verified.')
    st.write('Validation macro F1 selected the model. Audit groups are excluded from revised fits and selection. Earlier development may have exposed some articles, and earlier test results were already viewed.')
    st.link_button('Open training notebook', 'https://colab.research.google.com/drive/1Plj_ClnBPtqoSGWoWiQWmAhl1xUmP5PQ')
