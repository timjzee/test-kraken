from kraken.lib.train import RecognitionModel, KrakenTrainer
import glob

ground_truth = glob.glob('/vol/tensusers/timzee/kraken/trainingfiles/*.xml')
training_files = ground_truth[250:] # training data is shuffled internally
evaluation_files = ground_truth[:250]
model = RecognitionModel(training_data=training_files, evaluation_data=evaluation_files, format_type='xml')
trainer = KrakenTrainer()
trainer.fit(model)
#model.save_model("/vol/tensusers/timzee/kraken/test.mlmodel")