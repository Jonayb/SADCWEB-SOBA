# Main method for running all tests.
#
# https://github.com/stefanvanberkum/CD-ABSC

import time

import nltk

import lcr_fine_tune
import lcr_model
import lcr_test
from config import *
from load_data import *
from ontology import OntReasoner

nltk.download('punkt')


def main(_):
    """
    Runs all specified tests.

    :param _:
    :return:
    """
    # After running: back-up results file and model in case of running the model to be saved.
    # It is recommended to turn on logging of the output and to back that up as well for debugging purposes.

    rest_rest = True  # Run and save LCR-Rot-hop++ restaurant model.
    rest_test = False  # Test all domains on the off-the-shelf restaurant model.
    lapt_lapt = False  # Run LCR-Rot-hop++ laptop model.
    book_book = False  # Run LCR-Rot-hop++ book model.
    small_small = False  # Run LCR-Rot-hop++ for hotel and electronics models.
    rest_lapt_lapt = False  # Run fine-tuning on restaurant model for laptop domain.
    rest_book_book = False  # Run fine-tuning on restaurant model for book domain.
    rest_small_small = False  # Run fine-tuning on restaurant model for hotel and electronics domains.
    rest_rest_cross = False  # Run cross-validation for restaurant domain.
    lapt_lapt_cross = False  # Run cross-validation for laptop domain.
    run_ontology = True  # Run ontology reasoner.
    write_result = True  # Write results to text file.
    n_iter = 150

    FLAGS.n_iter = n_iter

    if rest_rest:
        # Run and save restaurant-restaurant.
        run_regular(domain="restaurant", year=2014, learning_rate=0.01, keep_prob=0.4, momentum=0.9, l2_reg=0.0001,
                    run_ontology=run_ontology, write_result=write_result, savable=True)

    if rest_test:
        laptop_domain = ["laptop", 2014]
        book_domain = ["book", 2019]
        hotel_domain = ["hotel", 2015]
        apex_domain = ["Apex", 2004]
        camera_domain = ["Camera", 2004]
        creative_domain = ["Creative", 2004]
        nokia_domain = ["Nokia", 2004]
        domains = [laptop_domain, book_domain, hotel_domain, apex_domain, camera_domain, creative_domain, nokia_domain]

        for domain in domains:
            # Run target data through restaurant model.
            run_test(source_domain="restaurant", source_year=2014, target_domain=domain[0], target_year=domain[1],
                     run_ontology=run_ontology, write_result=write_result)

    if lapt_lapt:
        # Run laptop-laptop for all splits.
        run_split(domain="laptop", year=2014, splits=9, split_size=250, learning_rate=0.07, keep_prob=0.7,
                  momentum=0.85, l2_reg=0.0001, run_ontology=run_ontology, write_result=write_result)

    if book_book:
        # Run book-book for all splits.
        run_split(domain="book", year=2019, splits=9, split_size=300, learning_rate=0.005, keep_prob=0.7, momentum=0.95,
                  l2_reg=0.01, run_ontology=run_ontology, write_result=write_result)

    if small_small:
        # Hyper parameters (learning_rate, keep_prob, momentum, l2_reg).
        hyper_hotel = [0.06, 0.4, 0.85, 0.00001]
        hyper_apex = [0.1, 0.7, 0.85, 0.1]
        hyper_camera = [0.02, 0.7, 0.85, 0.001]
        hyper_creative = [0.08, 0.7, 0.9, 0.1]
        hyper_nokia = [0.09, 0.6, 0.85, 0.01]

        # Name, year, train size step, hyperparameters.
        hotel_domain = ["hotel", 2015, 20, hyper_hotel]
        apex_domain = ["Apex", 2004, 25, hyper_apex]
        camera_domain = ["Camera", 2004, 31, hyper_camera]
        creative_domain = ["Creative", 2004, 54, hyper_creative]
        nokia_domain = ["Nokia", 2004, 22, hyper_nokia]
        domains = [hotel_domain, apex_domain, camera_domain, creative_domain, nokia_domain]

        # Run all single run models.
        for domain in domains:
            run_split(domain=domain[0], year=domain[1], splits=10, split_size=domain[2], learning_rate=domain[3][0],
                      keep_prob=domain[3][1], momentum=domain[3][2], l2_reg=domain[3][3], run_ontology=run_ontology,
                      write_result=write_result)

    if rest_lapt_lapt:
        # Run laptop-laptop fine-tuning on restaurant model.
        run_fine_tune(original_domain="restaurant", target_domain="laptop", year=2014, splits=9, split_size=250,
                      learning_rate=0.02, keep_prob=0.6, momentum=0.85, l2_reg=0.00001, run_ontology=run_ontology,
                      write_result=write_result)

    if rest_book_book:
        # Run book-book fine-tuning on restaurant model.
        run_fine_tune(original_domain="restaurant", target_domain="book", year=2019, splits=9, split_size=300,
                      learning_rate=0.01, keep_prob=0.6, momentum=0.85, l2_reg=0.001, run_ontology=run_ontology,
                      write_result=write_result)

    if rest_small_small:
        # Hyper parameters (learning_rate, keep_prob, momentum, l2_reg).
        hyper_hotel = [0.1, 0.5, 0.85, 0.1]
        hyper_apex = [0.005, 0.4, 0.95, 0.00001]
        hyper_camera = [0.02, 0.4, 0.95, 0.001]
        hyper_creative = [0.01, 0.3, 0.95, 0.01]
        hyper_nokia = [0.02, 0.3, 0.95, 0.001]

        # Name, year, train size step, hyperparameters.
        hotel_domain = ["hotel", 2015, 20, hyper_hotel]
        apex_domain = ["Apex", 2004, 25, hyper_apex]
        camera_domain = ["Camera", 2004, 31, hyper_camera]
        creative_domain = ["Creative", 2004, 54, hyper_creative]
        nokia_domain = ["Nokia", 2004, 22, hyper_nokia]
        domains = [hotel_domain, apex_domain, camera_domain, creative_domain, nokia_domain]

        # Run fine-tuning on restaurant model for all single run models.
        for domain in domains:
            run_fine_tune(original_domain="restaurant", target_domain=domain[0], year=domain[1], splits=10,
                          split_size=domain[2], learning_rate=domain[3][0], keep_prob=domain[3][1],
                          momentum=domain[3][2], l2_reg=domain[3][3], run_ontology=run_ontology,
                          write_result=write_result)

    if rest_rest_cross:
        # Run restaurant cross-validation.
        run_cross_val(source_domain="restaurant", source_year=2016, n_folds=10, learning_rate=0.001, keep_prob=0.7,
                      momentum=0.85, l2_reg=0.00001, run_ontology=run_ontology, write_result=write_result)

    if lapt_lapt_cross:
        # Run laptop cross-validation.
        run_cross_val(source_domain="laptop", source_year=2014, n_folds=10, learning_rate=0.001, keep_prob=0.7,
                      momentum=0.85, l2_reg=0.00001, run_ontology=run_ontology, write_result=write_result)

    print('Finished program successfully.')


def run_regular(domain, year, learning_rate, keep_prob, momentum, l2_reg, run_ontology, write_result, savable):
    """
    Run regular LCR-Rot-hop++ model which can be saved for fine-tuning.

    :param domain: train and test set domain
    :param year: train and test set year
    :param learning_rate: learning rate hyperparameter
    :param keep_prob: keep probability hyperparameter
    :param momentum: momentum hyperparameter
    :param l2_reg: l2 regularization hyperparameter
    :param run_ontology: True is ontology reasoner should be run before neural network
    :param write_result: True if results are to be saved to a text file
    :param savable: True if the model weights and biases are to be saved
    :return:
    """
    set_hyper_flags(learning_rate=learning_rate, keep_prob=keep_prob, momentum=momentum, l2_reg=l2_reg)
    set_other_flags(source_domain=domain, source_year=year, target_domain=domain, target_year=year)

    if write_result:
        FLAGS.results_file = "data/programGeneratedData/" + str(
            FLAGS.embedding_dim) + "results_" + domain + "_" + FLAGS.target_domain + "_test_" + str(
            FLAGS.year) + ".txt"
        with open(FLAGS.results_file, "w") as results:
            results.write(FLAGS.source_domain + " to " + FLAGS.target_domain + "\n---\n")
        FLAGS.writable = 1

    start_time = time.time()

    _, test_size, _, _ = load_data_and_embeddings(FLAGS, False)
    accuracy_ont = 1.0
    total_test_size = test_size

    # Run ontology reasoner.
    if run_ontology:
        accuracy_ont, remaining_size = run_ont()
        classified = test_size - remaining_size
        test_size = remaining_size
        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write("Ontology. Aspects classified: " + str(classified) + ", Accuracy: " + str(
                    accuracy_ont) + "\n---\n")

    # Run LCR-Rot-hop++.
    if savable:
        FLAGS.savable = 1
    lcr_model.main(FLAGS.train_path, FLAGS.test_path, accuracy_ont, total_test_size, test_size, FLAGS.learning_rate,
                   FLAGS.keep_prob1, FLAGS.momentum, FLAGS.l2_reg)
    tf.reset_default_graph()
    FLAGS.savable = 0

    end_time = time.time()
    run_time = end_time - start_time
    if write_result:
        with open(FLAGS.results_file, "a") as results:
            results.write("Runtime: " + str(run_time) + " seconds.\n\n")


def run_test(source_domain, source_year, target_domain, target_year, run_ontology, write_result):
    """
    Run the test data through the pre-trained model from the original domain.

    :param source_domain: the domain of the pre-trained model (always restaurant for this adaptation)
    :param source_year: the year of the pre-trained model domain dataset
    :param target_domain: the target domain
    :param target_year: the year of the target domain dataset
    :param run_ontology: True is ontology reasoner should be run before neural network
    :param write_result: True if results are to be saved to a text file
    :return:
    """
    set_other_flags(source_domain=source_domain, source_year=source_year, target_domain=target_domain,
                    target_year=target_year)

    if write_result:
        FLAGS.results_file = "data/programGeneratedData/" + str(
            FLAGS.embedding_dim) + "results_" + source_domain + "_" + FLAGS.target_domain + "_test_" + str(
            FLAGS.year) + ".txt"
        with open(FLAGS.results_file, "w") as results:
            results.write(source_domain + " to " + FLAGS.target_domain + "\n---\n")
        FLAGS.writable = 1

    start_time = time.time()

    _, test_size, _, _ = load_data_and_embeddings(FLAGS, False)
    accuracy_ont = 1.0

    # Run ontology reasoner.
    if run_ontology:
        accuracy_ont, remaining_size = run_ont()
        classified = test_size - remaining_size
        test_size = remaining_size
        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write("Ontology. Aspects classified: " + str(classified) + ", Accuracy: " + str(
                    accuracy_ont) + "\n---\n")

    # Run test method.
    lcr_test.main(FLAGS.test_path, accuracy_ont, test_size, test_size)
    tf.reset_default_graph()

    end_time = time.time()
    run_time = end_time - start_time
    if write_result:
        with open(FLAGS.results_file, "a") as results:
            results.write("Runtime: " + str(run_time) + " seconds.\n\n")


def run_split(domain, year, splits, split_size, learning_rate, keep_prob, momentum, l2_reg, run_ontology, write_result):
    """
    Runs regular LCR-Rot-hop++ for multiple cumulative training splits.

    :param domain: the domain
    :param year: the year of the domain dataset
    :param splits: the number of cumulative training splits
    :param split_size: the incremental size for each training split
    :param learning_rate: learning rate hyperparameter
    :param keep_prob: keep probability hyperparameter
    :param momentum: momentum hyperparameter
    :param l2_reg: l2 regularization hyperparameter
    :param run_ontology: True is ontology reasoner should be run before neural network
    :param write_result: True if results are to be saved to a text file
    :return:
    """
    set_hyper_flags(learning_rate=learning_rate, keep_prob=keep_prob, momentum=momentum, l2_reg=l2_reg)
    set_other_flags(source_domain=domain, source_year=year, target_domain=domain, target_year=year)

    if write_result:
        FLAGS.results_file = "data/programGeneratedData/" + str(
            FLAGS.embedding_dim) + "results_" + FLAGS.source_domain + "_" + FLAGS.target_domain + "_" + str(
            FLAGS.year) + ".txt"
        with open(FLAGS.results_file, "w") as results:
            results.write("")
        FLAGS.writable = 1

    for i in range(1, splits + 1):
        start_time = time.time()
        print("Running " + FLAGS.source_domain + " to " + FLAGS.target_domain + " using " + str(
            split_size * i) + " aspects...")

        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write(FLAGS.source_domain + " to " + FLAGS.target_domain + " using " + str(
                    split_size * i) + " aspects\n---\n")

        FLAGS.train_path = "data/programGeneratedData/BERT/" + FLAGS.source_domain + "/" + str(
            FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_train_" + str(FLAGS.year) + "_BERT_" + str(
            split_size * i) + ".txt"

        _, test_size, _, _ = load_data_and_embeddings(FLAGS, False)
        accuracy_ont = 1.0

        # Run ontology reasoner.
        if run_ontology:
            accuracy_ont, remaining_size = run_ont()
            classified = test_size - remaining_size
            test_size = remaining_size
            if FLAGS.writable == 1:
                with open(FLAGS.results_file, "a") as results:
                    results.write("Ontology. Aspects classified: " + str(classified) + ", Accuracy: " + str(
                        accuracy_ont) + "\n---\n")

        # Run LCR-Rot-hop++.
        lcr_model.main(FLAGS.train_path, FLAGS.test_path, accuracy_ont, test_size, test_size, FLAGS.learning_rate,
                       FLAGS.keep_prob1, FLAGS.momentum, FLAGS.l2_reg)
        tf.reset_default_graph()

        end_time = time.time()
        run_time = end_time - start_time
        if write_result:
            with open(FLAGS.results_file, "a") as results:
                results.write("Runtime: " + str(run_time) + " seconds.\n\n")

        FLAGS.test_path = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
            FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_test_" + str(FLAGS.year) + "_BERT.txt"


# Fine-tune method must be slightly adapted to work on original domains other than restaurant.
def run_fine_tune(original_domain, target_domain, year, splits, split_size, learning_rate, keep_prob, momentum, l2_reg,
                  run_ontology, write_result, split=True):
    """
    Runs fine-tuning on a model originally trained on another domain to adapt for cross-domain use.

    :param original_domain: the domain of the pre-trained model (always restaurant for this adaptation)
    :param target_domain: the target domain
    :param year: the year of the target domain dataset
    :param splits: the number of cumulative training splits
    :param split_size: the incremental size for each training split
    :param learning_rate: learning rate hyperparameter
    :param keep_prob: keep probability hyperparameter
    :param momentum: momentum hyperparameter
    :param l2_reg: l2 regularization hyperparameter
    :param run_ontology: True is ontology reasoner should be run before neural network
    :param write_result: True if results are to be saved to a text file
    :param split: True if the dataset is split in multiple cumulative training splits
    :return:
    """
    set_hyper_flags(learning_rate=learning_rate, keep_prob=keep_prob, momentum=momentum, l2_reg=l2_reg)
    set_other_flags(source_domain=target_domain, source_year=year, target_domain=target_domain, target_year=year)

    if write_result:
        FLAGS.results_file = "data/programGeneratedData/" + str(
            FLAGS.embedding_dim) + "results_" + original_domain + "_" + FLAGS.source_domain + "_" + FLAGS.target_domain + "_" + str(
            FLAGS.year) + ".txt"
        with open(FLAGS.results_file, "w") as results:
            results.write("")
        FLAGS.writable = 1

    if split:
        for i in range(1, splits + 1):
            start_time = time.time()
            print(
                "Running " + original_domain + " model with " + FLAGS.source_domain + " fine-tuning to " + FLAGS.target_domain + " using " + str(
                    split_size * i) + " aspects...")

            if FLAGS.writable == 1:
                with open(FLAGS.results_file, "a") as results:
                    results.write(
                        original_domain + " to " + FLAGS.target_domain + " with " + FLAGS.source_domain + " fine-tuning using " + str(
                            split_size * i) + " aspects\n---\n")

            FLAGS.train_path = "data/programGeneratedData/BERT/" + FLAGS.source_domain + "/" + str(
                FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_train_" + str(FLAGS.year) + "_BERT_" + str(
                split_size * i) + ".txt"

            _, test_size, _, _ = load_data_and_embeddings(FLAGS, False)
            accuracy_ont = 1.0

            # Run ontology reasoner.
            if run_ontology:
                accuracy_ont, remaining_size = run_ont()
                classified = test_size - remaining_size
                test_size = remaining_size
                if FLAGS.writable == 1:
                    with open(FLAGS.results_file, "a") as results:
                        results.write("Ontology. Aspects classified: " + str(classified) + ", Accuracy: " + str(
                            accuracy_ont) + "\n---\n")

            # Run fine-tuning method.
            lcr_fine_tune.main(FLAGS.train_path, FLAGS.test_path, accuracy_ont, test_size, test_size,
                               FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.momentum, FLAGS.l2_reg)
            tf.reset_default_graph()

            end_time = time.time()
            run_time = end_time - start_time
            if write_result:
                with open(FLAGS.results_file, "a") as results:
                    results.write("Runtime: " + str(run_time) + " seconds.\n\n")
    else:
        start_time = time.time()
        FLAGS.train_path = "data/programGeneratedData/BERT/" + FLAGS.source_domain + "/" + str(
            FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_train_" + str(FLAGS.year) + "_BERT.txt"
        print(
            "Running " + original_domain + " model with " + FLAGS.source_domain + " fine-tuning to " + FLAGS.target_domain + "...")

        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write(
                    original_domain + " to " + FLAGS.target_domain + " with " + FLAGS.source_domain + " fine-tuning\n---\n")

        _, test_size, _, _ = load_data_and_embeddings(FLAGS, False)
        accuracy_ont = 1.0

        # Run ontology reasoner.
        if run_ontology:
            accuracy_ont, remaining_size = run_ont()
            classified = test_size - remaining_size
            test_size = remaining_size
            if FLAGS.writable == 1:
                with open(FLAGS.results_file, "a") as results:
                    results.write("Ontology. Aspects classified: " + str(classified) + ", Accuracy: " + str(
                        accuracy_ont) + "\n---\n")

        # Run fine-tuning method.
        lcr_fine_tune.main(FLAGS.train_path, FLAGS.test_path, accuracy_ont, test_size, test_size, FLAGS.learning_rate,
                           FLAGS.keep_prob1, FLAGS.momentum, FLAGS.l2_reg)
        tf.reset_default_graph()

        end_time = time.time()
        run_time = end_time - start_time
        if write_result:
            with open(FLAGS.results_file, "a") as results:
                results.write("Runtime: " + str(run_time) + " seconds.\n\n")


def set_hyper_flags(learning_rate, keep_prob, momentum, l2_reg):
    """
    Sets hyperparameter flags.

    :param learning_rate: learning rate hyperparameter
    :param keep_prob: keep probability hyperparameter
    :param momentum: momentum hyperparameter
    :param l2_reg: l2 regularization hyperparameter
    :return:
    """
    FLAGS.learning_rate = learning_rate
    FLAGS.keep_prob1 = keep_prob
    FLAGS.keep_prob2 = keep_prob
    FLAGS.momentum = momentum
    FLAGS.l2_reg = l2_reg


def set_other_flags(source_domain, source_year, target_domain, target_year):
    """
    Set other flags.

    :param source_domain: the source domain
    :param source_year: the year of the source domain dataset
    :param target_domain: the target domain
    :param target_year: the year of the target domain dataset
    :return:
    """
    FLAGS.source_domain = source_domain
    FLAGS.target_domain = target_domain
    FLAGS.year = target_year
    FLAGS.train_path = "data/programGeneratedData/BERT/" + FLAGS.source_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_train_" + str(source_year) + "_BERT.txt"
    FLAGS.test_path = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_test_" + str(FLAGS.year) + "_BERT.txt"
    FLAGS.train_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.source_domain + "_" + str(
        source_year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.test_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
        FLAGS.year) + "_" + str(FLAGS.embedding_dim) + ".txt"


def run_ont():
    """
    Runs an ontology reasoner.
    NOTE. Not used and therefore not thoroughly tested in our research.
    :return:
    """
    print('Starting ontology reasoner...')
    ont = OntReasoner()
    accuracy_ont, remaining_size = ont.run(use_backup=True, path=FLAGS.test_path, use_svm=False)
    FLAGS.test_path = FLAGS.remaining_test_path
    return accuracy_ont, remaining_size


def run_cross_val(source_domain, source_year, n_folds, learning_rate, keep_prob, momentum, l2_reg, run_ontology,
                  write_result):
    """
    Runs LCR-Rot-hop++ cross-validation.
    NOTE. Not used and therefore not thoroughly tested in our research.

    :param source_domain: the source domain
    :param source_year: the year of the source domain dataset
    :param n_folds: number of folds for k-fold cross-validation
    :param learning_rate: learning rate hyperparameter
    :param keep_prob: keep probability hyperparameter
    :param momentum: momentum hyperparameter
    :param l2_reg: l2 regularization hyperparameter
    :param run_ontology: True is ontology reasoner should be run before neural network
    :param write_result: True if results are to be saved to a text file
    :return:
    """
    set_hyper_flags(learning_rate=learning_rate, keep_prob=keep_prob, momentum=momentum, l2_reg=l2_reg)
    set_other_flags(source_domain=source_domain, source_year=source_year, target_domain=source_domain,
                    target_year=source_year)

    if write_result:
        FLAGS.results_file = "data/programGeneratedData/" + str(
            FLAGS.embedding_dim) + "results_cross_" + FLAGS.source_domain + "_" + FLAGS.target_domain + "_" + str(
            FLAGS.year) + ".txt"
        with open(FLAGS.results_file, "w") as results:
            results.write("")
        FLAGS.writable = 1

    _, test_sizes, _, _ = load_cross_validation(FLAGS, n_folds, True)

    for i in range(1, n_folds + 1):
        start_time = time.time()
        print("Running " + FLAGS.source_domain + " to " + FLAGS.target_domain + " using fold " + str(i) + "...")

        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write(FLAGS.source_domain + " to " + FLAGS.target_domain + " using fold " + str(i) + "\n---\n")

        FLAGS.train_path = "data/programGeneratedData/crossValidation/cross_train_" + str(i) + ".txt"
        FLAGS.test_path = "data/programGeneratedData/crossValidation/cross_val_" + str(i) + ".txt"
        test_size = test_sizes[i-1]
        total_test_size = test_size
        accuracy_ont = 1.0

        # Run ontology reasoner.
        if run_ontology:
            accuracy_ont, remaining_size = run_ont()
            classified = test_size - remaining_size
            test_size = remaining_size
            if FLAGS.writable == 1:
                with open(FLAGS.results_file, "a") as results:
                    results.write("Ontology. Aspects classified: " + str(classified) + ", Accuracy: " + str(
                        accuracy_ont) + "\n---\n")

        # Run LCR-Rot-hop++.
        lcr_model.main(FLAGS.train_path, FLAGS.test_path, accuracy_ont, total_test_size, test_size, FLAGS.learning_rate,
                       FLAGS.keep_prob1, FLAGS.momentum, FLAGS.l2_reg)
        tf.reset_default_graph()

        end_time = time.time()
        run_time = end_time - start_time
        if write_result:
            with open(FLAGS.results_file, "a") as results:
                results.write("Runtime: " + str(run_time) + " seconds.\n\n")


if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()
