import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = GradientBoostingClassifier(learning_rate=1.0, max_depth=3, max_features=0.05, min_samples_leaf=15, min_samples_split=14, subsample=0.55)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
