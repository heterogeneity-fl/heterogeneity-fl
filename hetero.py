
def compute_grad_dissimilarity(net):
    """
    Two options:
    1. Only compute gradient dissimilarity during the first step of each round. This
    means that the global gradient is equal to the average local gradients at each
    client model.
    2. At every local step: For each client, broadcast the local model to all other
    clients, then compute local gradients at this local model. Compare local gradients
    to global gradient.

    Add other arguments?
    """

    # TODO: Start with option 1.
    pass

    # TODO: Average local gradients to compute global gradient.
    pass

    # TODO: Compute norm of difference between local and global gradient (for each client).
    pass

    # TODO: Save norm of difference to file.
    pass
