import pickle as pkl


def load_data(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)

    num_user = data['num_user']
    num_item = data['num_item']
    train_edges = data['train_edges']
    test_edges = data['test_edges']
    pe_user = data['positional_encoding']['user']
    pe_item = data['positional_encoding']['item']

    return num_user, num_item, train_edges, test_edges, pe_user, pe_item
