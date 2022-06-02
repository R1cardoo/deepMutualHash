import torch


def mean_average_precision(query_code,
                           database_code,
                           query_labels,
                           database_labels,
                           device,
                           topk=None,
                           ):
    """
    Calculate mean average precision(map).

    Args:
        query_code (torch.Tensor): Query data hash code.
        database_code (torch.Tensor): Database data hash code.
        query_labels (torch.Tensor): Query data targets, one-hot
        database_labels (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.

    Returns:
        meanAP (float): Mean Average Precision.
    """
    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP


def calculate_hamming(a, b):
    size = a.size(dim=0)
    hamming = 0
    for i in range(size):
        if a[i] != b[i]:
            hamming += 1
    return hamming


def acurracy(query_code,
             database_code,
             query_labels,
             database_labels,
             device,
             topk=None,
             ):
    correct_num = 0
    num_query = query_labels.shape[0]
    num_pos = 0
    num_neg = 0
    correct_pos = 0
    correct_neg = 0
    # num_database = database_labels.shape[0]
    for i in range(num_query):
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance
        retrieval = database_labels[torch.argmin(hamming_dist)]
        '''
                for j in range(num_database):
            hamming = calculate_hamming(query_code[i], database_code[j])
            if hamming < match_hamming:
                match_label = database_labels[j]
                match_hamming = hamming
        '''
        if query_labels[i].equal(torch.tensor([1., 0.]).to(device)):
            num_pos = num_pos + 1
        else: num_neg = num_neg + 1

        if retrieval.equal(query_labels[i]):
            correct_num = correct_num + 1
            if retrieval.equal(torch.tensor([1., 0.]).to(device)):
                correct_pos = correct_pos + 1
            else:
                correct_neg = correct_neg + 1

    acc = correct_num / num_query
    sen = correct_pos / num_pos
    spc = correct_neg / num_neg
    return acc, sen, spc
