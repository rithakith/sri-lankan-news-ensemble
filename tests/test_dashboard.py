"""Tests for model ordering, frozen evaluation, and dashboard states. ChatGPT-assisted."""
import sys
import json
import hashlib
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dashboard'))
sys.path.insert(0, str(ROOT / 'scripts'))
from core import MODEL_NAMES, model_text, infer, blend, validate_bundle, scores, category_scores, predictions
from import_dashboard_export import install


class FixedModel:
    def __init__(self, values):
        self.classes_ = np.arange(2)
        self.values = values
    def predict_proba(self, texts):
        return np.tile(self.values, (len(texts), 1))


class MetaModel(FixedModel):
    def predict_proba(self, blocks):
        # Require saved non-default base order: XGB, LR, RF.
        np.testing.assert_allclose(blocks[0], [.2,.8,.6,.4,.7,.3])
        return np.tile(self.values, (len(blocks),1))


@pytest.fixture
def bundle():
    from sklearn.preprocessing import LabelEncoder
    return {'ann_pipeline': FixedModel([.1,.9]),
            'stack_base_pipelines': {'LR':FixedModel([.6,.4]),'RF':FixedModel([.7,.3]),'XGB':FixedModel([.2,.8])},
            'stack_meta': MetaModel([.8,.2]), 'base_order':['XGB','LR','RF'],
            'label_encoder':LabelEncoder().fit(['business','sport']), 'stack_weight':.25,
            'input_policy':'headline_plus_first_200_body_words'}


def test_saved_base_order_and_hybrid(bundle):
    result, timing = infer(bundle,['Headline\nBody'])
    np.testing.assert_allclose(result['Selected hybrid'], [[.275,.725]])
    np.testing.assert_allclose(blend(result['Stacking'],result['ANN'],0), result['ANN'])
    assert set(timing)==set(MODEL_NAMES)
    bundle['ann_pipeline'].classes_ = np.array([1,0])
    with pytest.raises(ValueError, match='Class order'): validate_bundle(bundle)


def test_input_policy_and_unknown_weight():
    body=' '.join(str(i) for i in range(210))
    assert model_text('Headline', body)=='Headline\n'+' '.join(str(i) for i in range(200))
    with pytest.raises(ValueError): blend(np.ones(2),np.ones(2),float('nan'))


def frame():
    return pd.DataFrame({'doc_id':['1','2'],'headline':['A','B'],'BART_reference':['business','sport'],
                         **{name:['business','business'] for name in MODEL_NAMES}})


def test_metrics_include_unsupported_category():
    f=frame()
    categories=['business','sport','education']
    result=scores(f,categories)
    assert result.Agreement.eq(.5).all()
    assert result['Macro-F1'].iloc[0] == pytest.approx(2/9)
    detail=category_scores(f,'LR',categories)
    assert np.isnan(detail.loc[2,'F1'])


def test_prediction_export_rejects_unknown_category(tmp_path):
    f=frame()
    f.to_csv(tmp_path/'test_predictions.csv',index=False)
    assert len(predictions(tmp_path,'test',['business','sport']))==2
    with pytest.raises(ValueError): predictions(tmp_path,'test',['business'])


def test_import_rejects_unsafe_paths(tmp_path):
    z=tmp_path/'bad.zip'
    with zipfile.ZipFile(z,'w') as f: f.writestr('../outside','bad')
    with pytest.raises(ValueError,match='unexpected'): install(z,tmp_path/'out',False)


def test_app_all_empty_states(monkeypatch,tmp_path):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('NEWS_ARTIFACT_DIR',str(tmp_path))
    app=AppTest.from_file(str(ROOT/'dashboard/app.py'),default_timeout=20).run()
    assert not app.exception
    for page in ['Classify an article','Compare models','Explore errors','Hybrid lab','Methodology','Overview']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
    app.sidebar.radio[0].set_value('Classify an article').run()
    app.button[0].click().run()
    assert len(app.warning)>0


def test_app_evaluation_views(monkeypatch,tmp_path):
    from streamlit.testing.v1 import AppTest
    f=frame()
    f.to_csv(tmp_path/'validation_predictions.csv',index=False)
    f.assign(doc_id=['3','4']).to_csv(tmp_path/'test_predictions.csv',index=False)
    run={'mode':'pilot_v7','taxonomy_version':'v7','categories':['business','sport','education'],
         'input_policy':'headline_plus_first_200_body_words','stack_weight':0,
         'comparison_counts':{'train':10,'validation':2,'test':2}}
    (tmp_path/'run.json').write_text(json.dumps(run))
    monkeypatch.setenv('NEWS_ARTIFACT_DIR',str(tmp_path))
    app=AppTest.from_file(str(ROOT/'dashboard/app.py'),default_timeout=20).run()
    assert not app.exception
    for page in ['Compare models','Explore errors','Hybrid lab','Methodology']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page


def test_installed_model_live_prediction_and_slider(monkeypatch):
    import joblib
    from streamlit.testing.v1 import AppTest
    artifacts=ROOT/'dashboard/artifacts'
    if not (artifacts/'hybrid_bundle.joblib').exists():
        pytest.skip('No installed trained bundle')
    bundle=joblib.load(artifacts/'hybrid_bundle.joblib')
    text=model_text('Sri Lanka announces new education programme',
                    'Schools will receive new learning materials and teacher training this year.')
    result,_=infer(bundle,[text])
    for probs in result.values():
        assert probs.shape==(1,len(bundle['label_encoder'].classes_))
        np.testing.assert_allclose(probs.sum(axis=1),1,atol=1e-5)
    before=(artifacts/'test_predictions.csv').read_bytes()
    monkeypatch.setenv('NEWS_ARTIFACT_DIR',str(artifacts))
    app=AppTest.from_file(str(ROOT/'dashboard/app.py'),default_timeout=30).run()
    app.sidebar.radio[0].set_value('Classify an article').run()
    app.text_input[0].set_value('Sri Lanka announces new education programme')
    app.text_area[0].set_value('Schools will receive new learning materials and teacher training this year.')
    app.button[0].click().run()
    assert not app.exception
    assert len(app.dataframe)>=1
    assert 'prediction' in app.session_state
    app.sidebar.radio[0].set_value('Hybrid lab').run()
    app.slider[0].set_value(.5).run()
    assert not app.exception
    assert before==(artifacts/'test_predictions.csv').read_bytes()
