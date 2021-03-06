
import kfp
import kfp.dsl as dsl

@dsl.pipeline(name='Train Pipeline', description='Load patient data and trains the LSTM model.')
def train_pipeline():
    # Loads the yaml manifest for each component
    load_data = kfp.components.load_component_from_file('kubeflow-pipeline/load_data/load_data.yaml')
    train = kfp.components.load_component_from_file('kubeflow-pipeline/train/train.yaml')
    #logistic_regression = kfp.components.load_component_from_file('logistic_regression/logistic_regression.yaml')

    # Run download_data task
    load_task = load_data('540')

    # Run tasks "decison_tree" and "logistic_regression" given
    # the output generated by "download_task".
    train_task = train('540', 'models')
    #logistic_regression_task = logistic_regression(download_task.output)

    # Given the outputs from "decision_tree" and "logistic_regression"
    # the component "show_results" is called to print the results.
    #show_results(decision_tree_task.output, logistic_regression_task.output)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(train_pipeline, 'TrainPipeline.yaml')