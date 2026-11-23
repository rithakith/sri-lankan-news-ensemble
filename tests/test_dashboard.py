"""Verify real predictions, saved evaluations and dashboard interactions."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dashboard'))
from core import MODEL, STAGES, audit_matrix, load_classifier, predict_article, selection, table

def test_saved_metrics_reproduce_reviewed_predictions():
    audit = table('human_audit_joined_predictions.csv')
    assert len(audit) == 300 and audit.doc_id.is_unique
    mapping = {'BART': 'bart_label', 'Initial ensemble': 'initial_ensemble_prediction',
               'Refined ensemble': 'refined_ensemble_prediction', 'Final classifier': 'final_prediction'}
    for _, row in table('human_audit_metrics.csv').query("weighting == 'unweighted audit'").iterrows():
        assert accuracy_score(audit.human_label, audit[mapping[row.model]]) == pytest.approx(row.accuracy)
        assert f1_score(audit.human_label, audit[mapping[row.model]], labels=selection()['categories'],
                        average='macro', zero_division=0) == pytest.approx(row.macro_f1)
    for columns in [('human_label','final_prediction'), ('human_label','bart_label'), ('bart_label','final_prediction')]:
        cm = audit_matrix(*columns, selection()['categories'])
        assert cm.shape == (13, 13)
        np.testing.assert_allclose(cm.sum(axis=1), 1)

def test_all_training_stages_and_frozen_winner():
    compared = table('model_comparison_validation.csv')
    assert len(compared) == 20
    assert set(compared.stage) == set(STAGES)
    winner = compared.sort_values('macro_f1', ascending=False).iloc[0]
    assert (winner.model, winner.stage) == ('LinearSVC', 'selected_refined')

def test_real_pipeline_decodes_integer_classes_and_scores():
    saved = load_classifier()
    headline, body = 'Sri Lanka wins cricket match', 'The national cricket team won the tournament after scoring runs.'
    result = predict_article(saved, headline, body)
    expected = int(saved['pipeline'].predict([headline+'\n'+body])[0])
    assert result['category'] == saved['topics'][expected]
    assert result['known_terms'] > 0
    assert len(result['alternatives']) == 3
    assert result['alternatives']['Decision score'].is_monotonic_decreasing
    with pytest.raises(ValueError):
        predict_article(saved, '', ' ')

def test_dashboard_views_and_live_prediction():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / 'dashboard/app.py')).run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == '64%'
    next(r for r in app.radio if r.label == 'Evaluation').set_value('Validation experiments').run()
    assert not app.exception
    for choice in ['Human labels vs BART', 'LinearSVC vs BART']:
        app.selectbox[0].set_value(choice).run()
        assert not app.exception
    app.sidebar.radio[0].set_value('About the study').run()
    assert not app.exception
    app.sidebar.radio[0].set_value('Predict an article').run()
    app.button[0].click().run()
    assert app.warning
    app.text_input[0].set_value('Sri Lanka wins cricket match')
    app.text_area[0].set_value('The cricket team won the tournament after scoring runs.')
    app.button[0].click().run(timeout=30)
    assert not app.exception and len(app.dataframe) == 1
