# 11. Frozen-model test evaluation and plots.
# No fit(), tuning, epoch selection or blend-weight selection in this section.
test_ids = [i for i in assigned['test'] if rows[i]['predicted_topic'] in eligible]
test_text = [model_text(rows[i]) for i in test_ids]
test_targets = stack_encoder.transform([rows[i]['predicted_topic'] for i in test_ids])
test_predictions = {}
test_base_probs = []
for key, label in [('LR', 'LR'), ('RF', 'Random Forest'), ('XGB', 'XGBoost')]:
    model = fitted_bases[key]
    assert np.array_equal(model.classes_, np.arange(K))
    probs = model.predict_proba(test_text)
    test_base_probs.append(probs)
    test_predictions[label] = probs.argmax(axis=1)
test_stack_probs = stack_meta.predict_proba(np.hstack(test_base_probs))
test_ann_probs = ann_pipeline.predict_proba(test_text)
test_hybrid_probs = blend_weight * test_stack_probs + (1-blend_weight) * test_ann_probs
for probs in [test_stack_probs, test_ann_probs, test_hybrid_probs]:
    assert probs.shape == (len(test_ids), K)
    assert np.isfinite(probs).all() and np.allclose(probs.sum(axis=1), 1, atol=1e-5)
test_predictions.update({'Stacking': test_stack_probs.argmax(axis=1),
    'ANN': test_ann_probs.argmax(axis=1), 'Selected hybrid': test_hybrid_probs.argmax(axis=1)})
TEST_DIR = OUTPUT_DIR / 'test_analysis'
TEST_DIR.mkdir(exist_ok=True)
print(f'Frozen test evaluation: {len(test_ids)} / {len(assigned["test"])} test articles; {K} experiment categories.')
print('Excluded experiment categories:', excluded)
print(f'Fixed validation-selected hybrid: {blend_weight:.0%} stack + {1-blend_weight:.0%} ANN; fixed ANN epoch {best_epoch}')
print('BART pseudo-label agreement, not independently verified topic accuracy.')

test_metric_rows = []
test_f1 = []
test_reports = {}
for name in names:
    pred = test_predictions[name]
    test_metric_rows.append({'model': name, 'agreement': accuracy_score(test_targets, pred),
        'macro_f1': f1_score(test_targets, pred, labels=np.arange(K), average='macro', zero_division=0),
        'weighted_f1': f1_score(test_targets, pred, average='weighted', zero_division=0)})
    test_f1.append(precision_recall_fscore_support(test_targets, pred, labels=np.arange(K), zero_division=0)[2])
    test_reports[name] = classification_report(test_targets, pred, labels=np.arange(K),
        target_names=topic_names, output_dict=True, zero_division=0)
test_metric_table = pd.DataFrame(test_metric_rows).set_index('model')
display(test_metric_table.round(4))
test_support = np.bincount(test_targets, minlength=K)

fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
test_metric_table.plot.bar(ax=axes[0], rot=20, ylim=(0, 1))
axes[0].set(title=f'Test comparison: same {len(test_ids)} articles, frozen models', xlabel='Model', ylabel='Score (0–1)')
stages = pd.DataFrame({'validation': metric_table['macro_f1'], 'test': test_metric_table['macro_f1']})
stages.plot.bar(ax=axes[1], rot=20, ylim=(0, 1))
axes[1].set(title='Macro-F1: validation versus test', xlabel='Model', ylabel='Macro-F1 (all experiment categories)')
fig.savefig(TEST_DIR / '01_test_scores.png', dpi=160, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
test_f1 = np.array(test_f1)
masked = np.ma.masked_where(np.broadcast_to(test_support == 0, test_f1.shape), test_f1)
cmap = plt.get_cmap('Blues').copy()
cmap.set_bad('lightgray')
im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(K), [f'{t}\n(n={n})' for t, n in zip(topic_names, test_support)], rotation=40, ha='right')
ax.set_yticks(range(len(names)), names)
for r in range(len(names)):
    for c in range(K):
        ax.text(c, r, 'N/A' if test_support[c] == 0 else f'{test_f1[r,c]:.2f}',
            ha='center', va='center', color='white' if test_f1[r,c] > .6 and test_support[c] else 'black')
ax.set(title='Test F1 by category; n = test support', xlabel='BART reference category', ylabel='Model')
fig.colorbar(im, ax=ax, label='F1')
fig.savefig(TEST_DIR / '02_test_category_f1.png', dpi=160, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(3, 2, figsize=(12, 16), constrained_layout=True)
for ax, name in zip(axes.flat, names):
    cm = confusion_matrix(test_targets, test_predictions[name], labels=np.arange(K))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=max(test_support))
    ax.set_xticks(range(K), short_topics, rotation=90)
    ax.set_yticks(range(K), short_topics)
    ax.tick_params(labelsize=8)
    ax.set(title=name, xlabel='Predicted category', ylabel='BART reference category')
    for r, c in zip(*np.nonzero(cm)):
        ax.text(c, r, str(cm[r,c]), ha='center', va='center',
            color='white' if cm[r,c] > max(test_support)/2 else 'black')
fig.colorbar(im, ax=list(axes.flat), label='Test articles')
fig.suptitle('Test confusion matrices — shared category order and colour scale')
fig.savefig(TEST_DIR / '03_test_confusion.png', dpi=160, bbox_inches='tight')
plt.show()

test_disagreement = np.array([[np.mean(test_predictions[a] != test_predictions[b]) for b in names] for a in names])
fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
im = ax.imshow(test_disagreement, cmap='Oranges', vmin=0, vmax=1)
ax.set_xticks(range(len(names)), names, rotation=35, ha='right')
ax.set_yticks(range(len(names)), names)
for r in range(len(names)):
    for c in range(len(names)):
        ax.text(c, r, f'{test_disagreement[r,c]:.0%}', ha='center', va='center')
ax.set(title='Test prediction disagreement')
fig.colorbar(im, ax=ax, label='Fraction of different predictions')
fig.savefig(TEST_DIR / '04_test_disagreement.png', dpi=160, bbox_inches='tight')
plt.show()

test_inspection = pd.DataFrame({'doc_id': [rows[i]['doc_id'] for i in test_ids],
    'headline': [rows[i]['description'] for i in test_ids],
    'BART_reference': stack_encoder.inverse_transform(test_targets)})
for name in names:
    test_inspection[name] = stack_encoder.inverse_transform(test_predictions[name].astype(int))
display(test_inspection)
test_metric_table.to_csv(TEST_DIR / 'test_metrics.csv')
test_inspection.to_csv(TEST_DIR / 'test_article_predictions.csv', index=False)
stages.to_csv(TEST_DIR / 'validation_test_comparison.csv')
(TEST_DIR / 'classification_reports.json').write_text(json.dumps(test_reports, indent=2))
(TEST_DIR / 'evaluation_settings.json').write_text(json.dumps({'mode': MODE,
    'test_articles': len(test_ids), 'original_test_articles': len(assigned['test']),
    'categories': topic_names, 'excluded_categories': excluded,
    'stack_weight_fixed_from_validation': blend_weight, 'ann_epoch_fixed_from_validation': best_epoch,
    'no_test_tuning': True, 'reference': 'BART pseudo-labels',
    'warning': 'Labels are model-generated; no independent human-labelled benchmark is available.'}, indent=2))
print('One changed test prediction changes agreement by', f'{1/len(test_ids):.1%}')
print('Do not tune settings against these test results and continue calling them untouched test results.')
print('No hybrid improvement can be claimed if the frozen selected weight is an endpoint.')
