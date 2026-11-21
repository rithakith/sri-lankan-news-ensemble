# 10. Visual comparison and interpretation.
# All model comparisons below use the SAME stacking-experiment validation subset.
# https://scikit-learn.org/stable/modules/model_evaluation.html
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

VIS_DIR = OUTPUT_DIR / 'visual_analysis'
VIS_DIR.mkdir(exist_ok=True)
names = ['LR', 'Random Forest', 'XGBoost', 'Stacking', 'ANN', 'Selected hybrid']
predictions = {
    'LR': fitted_bases['LR'].predict(stack_valid_text),
    'Random Forest': fitted_bases['RF'].predict(stack_valid_text),
    'XGBoost': fitted_bases['XGB'].predict(stack_valid_text),
    'Stacking': stack_valid_pred, 'ANN': ann_pred, 'Selected hybrid': hybrid_pred,
}
assert all(len(p) == len(stack_valid_y) for p in predictions.values())
topic_names = stack_encoder.classes_.tolist()
print(f'Comparison scope: {len(stack_y)} training articles, {len(stack_valid_y)} validation articles, {K} categories.')
print('Excluded from this experiment:', excluded)
print('BART reference labels. Validation results are used for model selection.')

metric_rows = []
class_f1 = []
for name in names:
    pred = predictions[name]
    metric_rows.append({'model': name,
        'agreement': accuracy_score(stack_valid_y, pred),
        'macro_f1': f1_score(stack_valid_y, pred, labels=np.arange(K), average='macro', zero_division=0),
        'weighted_f1': f1_score(stack_valid_y, pred, average='weighted', zero_division=0)})
    class_f1.append(precision_recall_fscore_support(stack_valid_y, pred, labels=np.arange(K), zero_division=0)[2])
metric_table = pd.DataFrame(metric_rows).set_index('model')

# 1: Same-subset scores and training/validation category support.
fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
metric_table.plot.bar(ax=axes[0], ylim=(0, 1), rot=25)
axes[0].set(title='Validation agreement and F1 (same articles)', ylabel='Score (0–1)', xlabel='Model')
support = pd.DataFrame({'training': np.bincount(stack_y, minlength=K),
    'validation': np.bincount(stack_valid_y, minlength=K)}, index=topic_names)
support.plot.barh(ax=axes[1])
axes[1].set(title='Category support: imbalance and missing coverage', xlabel='Articles', ylabel='Category')
fig.savefig(VIS_DIR / '01_scores_and_support.png', dpi=160, bbox_inches='tight')
plt.show()

# 2: Per-category F1. Grey cells mean no reference examples, not measured failure.
f1_matrix = np.array(class_f1)
validation_support = np.bincount(stack_valid_y, minlength=K)
masked_f1 = np.ma.masked_where(np.broadcast_to(validation_support == 0, f1_matrix.shape), f1_matrix)
fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
cmap = plt.get_cmap('Blues').copy()
cmap.set_bad('lightgray')
im = ax.imshow(masked_f1, vmin=0, vmax=1, cmap=cmap, aspect='auto')
ax.set_xticks(np.arange(K), [f'{topic}\n(n={n})' for topic, n in zip(topic_names, validation_support)], rotation=35, ha='right')
ax.set_yticks(np.arange(len(names)), names)
for r in range(len(names)):
    for c in range(K):
        label = 'N/A' if validation_support[c] == 0 else f'{f1_matrix[r,c]:.2f}'
        ax.text(c, r, label, ha='center', va='center', color='white' if f1_matrix[r,c] > 0.6 and validation_support[c] else 'black')
ax.set(title='F1 by category — n is validation support', xlabel='BART category', ylabel='Model')
fig.colorbar(im, ax=ax, label='F1 (0–1)')
fig.savefig(VIS_DIR / '02_category_f1.png', dpi=160, bbox_inches='tight')
plt.show()

# 3: Confusion matrices with identical axes/scales across all models.
fig, axes = plt.subplots(2, 3, figsize=(16, 11), constrained_layout=True)
short_topics = [t.replace('_', '\n') for t in topic_names]
for ax, name in zip(axes.flat, names):
    cm = confusion_matrix(stack_valid_y, predictions[name], labels=np.arange(K))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=max(validation_support))
    ax.set_xticks(np.arange(K), short_topics, rotation=90)
    ax.set_yticks(np.arange(K), short_topics)
    ax.tick_params(labelsize=8)
    ax.set(title=name, xlabel='Predicted category', ylabel='BART reference category')
    for r, c in zip(*np.nonzero(cm)):
        ax.text(c, r, str(cm[r,c]), ha='center', va='center', color='white' if cm[r,c] > max(validation_support)/2 else 'black')
fig.colorbar(im, ax=list(axes.flat), label='Articles')
fig.suptitle('Where predictions go wrong: diagonal entries match BART')
fig.savefig(VIS_DIR / '03_confusion_matrices.png', dpi=160, bbox_inches='tight')
plt.show()

# 4: ANN optimisation and hybrid weight sensitivity.
hist = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
axes[0].plot(hist['epoch'], hist['training_loss'], label='Training objective (includes regularisation)')
axes[0].plot(hist['epoch'], hist['validation_log_loss'], label='Validation log loss')
axes[0].axvline(best_epoch, color='gray', linestyle='--', label=f'Selected epoch {best_epoch}')
axes[0].set(title='ANN learning history', xlabel='Epoch', ylabel='Loss (lower is better)')
axes[0].legend(fontsize=8)
axes[1].plot(blend_search['stack_weight'], blend_search['macro_f1'], marker='o', label='Macro-F1')
axes[1].plot(blend_search['stack_weight'], blend_search['validation_agreement'], marker='s', label='Agreement')
axes[1].axvline(blend_weight, color='gray', linestyle='--', label='Selected weight')
axes[1].set(title='Does adding the stack help the ANN?', xlabel='Stack weight (0 = ANN alone; 1 = stack alone)', ylabel='Validation score', ylim=(0, 1))
axes[1].legend(fontsize=8)
fig.savefig(VIS_DIR / '04_ann_learning_and_blending.png', dpi=160, bbox_inches='tight')
plt.show()

# 5: Diversity is useful only if models make complementary correct predictions.
disagreement = np.array([[np.mean(predictions[a] != predictions[b]) for b in names] for a in names])
fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
im = ax.imshow(disagreement, cmap='Oranges', vmin=0, vmax=1)
ax.set_xticks(range(len(names)), names, rotation=35, ha='right')
ax.set_yticks(range(len(names)), names)
for r in range(len(names)):
    for c in range(len(names)):
        ax.text(c, r, f'{disagreement[r,c]:.0%}', ha='center', va='center')
ax.set(title='Prediction disagreement on the same validation articles')
fig.colorbar(im, ax=ax, label='Fraction of different predictions')
fig.savefig(VIS_DIR / '05_model_disagreement.png', dpi=160, bbox_inches='tight')
plt.show()

stack_correct = stack_valid_pred == stack_valid_y
ann_correct = ann_pred == stack_valid_y
complementarity = pd.Series({
    'Both match BART': int((stack_correct & ann_correct).sum()),
    'Only ANN matches': int((~stack_correct & ann_correct).sum()),
    'Only stack matches': int((stack_correct & ~ann_correct).sum()),
    'Neither matches': int((~stack_correct & ~ann_correct).sum()),
}, name='Articles')
display(complementarity.to_frame())
inspection = pd.DataFrame({'doc_id': [rows[i]['doc_id'] for i in stack_valid_ids],
    'headline': [rows[i]['description'] for i in stack_valid_ids],
    'BART_reference': stack_encoder.inverse_transform(stack_valid_y)})
for name in names:
    inspection[name] = stack_encoder.inverse_transform(predictions[name].astype(int))
inspection['models_disagree'] = np.array(list(predictions.values())).max(axis=0) != np.array(list(predictions.values())).min(axis=0)
display(inspection)
inspection.to_csv(VIS_DIR / 'article_level_comparison.csv', index=False)
metric_table.to_csv(VIS_DIR / 'validation_metrics.csv')
support.to_csv(VIS_DIR / 'category_support.csv')
pd.DataFrame(f1_matrix, index=names, columns=topic_names).to_csv(VIS_DIR / 'category_f1.csv')

best_name = metric_table['macro_f1'].idxmax()
print('Evidence-based observations for this run:')
print(f'- Highest validation macro-F1: {best_name} ({metric_table.loc[best_name, "macro_f1"]:.3f}). This is not proof of superiority.')
print(f'- No validation examples for: {[t for t, n in zip(topic_names, validation_support) if n == 0]}.')
print(f'- One changed prediction moves agreement by {1/len(stack_valid_y):.1%}.')
print('- ANN epoch and hybrid weight were selected on this validation set; do not report those scores as independent test performance.')
print('- Inspect the confusion matrices for majority-category collapse; disagreement alone does not imply useful diversity.')
print('- Training/validation loss curves suggest optimisation behaviour, not a causal explanation or proof of generalisation.')
