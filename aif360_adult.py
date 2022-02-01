from aif360.datasets import AdultDataset

ds_orig = AdultDataset()
# Split into train and test partitions
ds_orig_tr, ds_orig_te = ds_orig.split([0.7], shuffle=True, seed=1)

# Look into the training dataset
print("Training Dataset shape")
print(ds_orig_tr.features.shape)
print("Favorable and unfavorable outcome labels")
print(ds_orig_tr.favorable_label, ds_orig_tr.unfavorable_label)
print("Metadata for labels")
print(ds_orig_tr.metadata["label_maps"])
print("Protected attribute names")
print(ds_orig_tr.protected_attribute_names)
print("Privileged and unprivileged protected attribute values")
print(ds_orig_tr.privileged_protected_attributes,
ds_orig_tr.unprivileged_protected_attributes)
print("Metadata for protected attributes")
print(ds_orig_tr.metadata["protected_attribute_maps"])

# Load the metric class
from aif360.metrics import BinaryLabelDatasetMetric

# Define privileged and unprivileged groups
priv = [{'sex': 1}] # Male
unpriv = [{'sex': 0}] # Female
# Create the metric object
metric_otr = BinaryLabelDatasetMetric( ds_orig_tr,
unprivileged_groups=unpriv, privileged_groups=priv)
# Load and create explainers
from aif360.explainers import MetricTextExplainer, MetricJSONExplainer
text_exp_otr = MetricTextExplainer(metric_otr)
json_exp_otr = MetricJSONExplainer(metric_otr)
# Print statistical parity difference
print(text_exp_otr.statistical_parity_difference())
print(json_exp_otr.statistical_parity_difference())

# Import the reweighing preprocessing algorithm class
from aif360.algorithms.preprocessing.reweighing import Reweighing
# Create the algorithm object
RW = Reweighing(unprivileged_groups=unpriv, privileged_groups=priv)
# Train and predict on the training data

# Uses scikit-learn convention (fit, predict, transform)
RW.fit(ds_orig_tr)
ds_transf_tr = RW.transform(ds_orig_tr)

# Create the metric object for pre-processed data
metric_ttr = BinaryLabelDatasetMetric(ds_transf_tr,
unprivileged_groups=unpriv, privileged_groups=priv)
# Create explainer
text_exp_ttr = MetricTextExplainer(metric_ttr)
# Print statistical parity difference
print(text_exp_ttr.statistical_parity_difference())

# Apply the learned re-weighing pre-processor
ds_transf_te = RW.transform(ds_orig_te)
# Create metric objects for original and
# pre-processed test data
metric_ote = BinaryLabelDatasetMetric(ds_orig_te,
unprivileged_groups=unpriv, privileged_groups=priv)
metric_tte = BinaryLabelDatasetMetric(ds_transf_te,
unprivileged_groups=unpriv, privileged_groups=priv)
# Create explainers for both metric objects
text_exp_ote = MetricTextExplainer(metric_ote)
text_exp_tte = MetricTextExplainer(metric_tte)
# Print statistical parity difference
print(text_exp_ote.statistical_parity_difference())
print(text_exp_tte.statistical_parity_difference())

test = ds_orig_te.convert_to_dataframe()[0]
test.to_csv('adult_aif360.test')

train = ds_orig_tr.convert_to_dataframe()[0]
train.to_csv('adult_aif360.train')