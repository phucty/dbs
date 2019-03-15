import io_worker
import argparse
import distances
from collections import defaultdict
from attribute_obj import NumericalAttribute


def get_labels(queries, knowledge_base, method, trans_size):
    model = None
    if "dsl" in method:
        model = io_worker.load_object_pickle("./data/dsl.model")
    if "dbs" in method:
        for attr in knowledge_base:
            attr.values = distances.get_transformation(attr.values, trans_size)
    results = []
    for q in queries:
        ranking = defaultdict(float)
        if "dbs" in method:
            query = distances.get_transformation(q.values, trans_size)
        else:
            query = q.values
        for attr in knowledge_base:
            if "semantictyper" in method:
                temp_dis = distances.distance_ks_test_p(query, attr.values)
            elif "dsl" in method:
                temp_dis = distances.distance_dsl(query, attr.values, model, distances.get_dsl_vector)
            elif "dbs1" in method:
                temp_dis = distances.distance_numerical(query, attr.values, "l1")
            elif "dbs2" in method:
                temp_dis = distances.distance_numerical(query, attr.values, "l2")
            else:
                temp_dis = distances.distance_numerical(query, attr.values, "inf")

            old_best_score = ranking.get(attr.label, 1E100)
            if temp_dis < old_best_score:
                ranking[attr.label] = temp_dis
        ranking = sorted(ranking.items(), key=lambda x:x[1])
        results.append((q, ranking[:10]))
    return method, results


dataset_names = ["all", "city", "dbpedia", "wikidata", "open"]
methods = ["semantictyper", "dsl", "dbs1", "dbs2", "dbsinf"]


def get_arguments():
    parser = argparse.ArgumentParser(description="Semantic Labeling for Numerical Values")
    parser.add_argument("--dataset", choices=dataset_names, default="all", help="Dataset in {}".format(dataset_names))
    parser.add_argument("--method", choices=methods, default="dbs1", help="Method in {}".format(methods))
    parser.add_argument("--trans_size", default=100, type=int, help="Distribution transformation size")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    io_worker.print_status(args)
    if "all" in args.dataset:
        process_datasets = dataset_names[1:]
    else:
        process_datasets = [args.dataset]

    labels = set()
    knowledge_base = []
    io_worker.print_status("Data Loading: ")
    for data_name in process_datasets:
        temp_labels, temp_kb = io_worker.load_numerical_dataset(data_name)
        labels = labels.union(temp_labels)
        knowledge_base.extend(temp_kb)
        io_worker.print_status("Overall: %d(labels) - %d(columns) | Load: %s: %d(labels) - %d(columns) " %
                               (len(labels), len(knowledge_base), data_name, len(temp_labels), len(temp_kb)))

    # Query:
    # Bag of Numbers
    query_1 = NumericalAttribute("Unknown", [1.7, 1.65, 1.7, 1.55, 1.71, 1.65, 1.88])
    query_2 = NumericalAttribute("Unknown", [0.88, 0.92, 0.17, 0.65, 0.66, 0.90, 0.88, 0.72, 0.76, 0.99])
    query_3 = NumericalAttribute("Unknown", [2018, 1965, 1987, 1999, 2017, 2015, 2011, 2012, 2012, 2011])

    # CSV files
    queries = io_worker.load_queries("test")
    queries = [query_1, query_2, query_3] + queries

    # Semantic labeling using the Manhattan distance DBS1 (l1)
    sem_method, sem_result = get_labels(queries, knowledge_base, "dbs1", args.trans_size)
    io_worker.print_labeling(sem_method, sem_result)

    # Compare mode between distribution-based distance and other p-value based method
    io_worker.print_labeling_compare_mode([get_labels(queries, knowledge_base, "dbs1", args.trans_size),
                                           get_labels(queries, knowledge_base, "dbs2", args.trans_size),
                                           get_labels(queries, knowledge_base, "dbsinf", args.trans_size),
                                           get_labels(queries, knowledge_base, "semantictyper", args.trans_size),
                                           get_labels(queries, knowledge_base, "dsl", args.trans_size),
                                           ])


