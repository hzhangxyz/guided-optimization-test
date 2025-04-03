import torch


@torch.jit.script
def _unit_term(left: int, left_term: torch.Tensor, middle: int, right_term: torch.Tensor, right: int) -> torch.Tensor:
    left_tensor = torch.diag(torch.ones([int(2**left)], dtype=torch.complex64))
    middle_tensor = torch.diag(torch.ones([int(2**middle)], dtype=torch.complex64))
    right_tensor = torch.diag(torch.ones([int(2**right)], dtype=torch.complex64))
    result = torch.kron(left_tensor, left_term)
    result = torch.kron(result, middle_tensor)
    result = torch.kron(result, right_term)
    result = torch.kron(result, right_tensor)
    return result


@torch.jit.script
def _get_index(site: tuple[int, int], size: tuple[int, int]) -> int:
    return site[0] * size[1] + site[1]


@torch.jit.script
def _single_term(
        index1: int,
        index2: int,
        total: int,
        _sigma_x: torch.Tensor = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64),
        _sigma_y: torch.Tensor = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64),
        _sigma_z: torch.Tensor = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64),
) -> torch.Tensor:
    x = _unit_term(index1, _sigma_x, index2 - index1 - 1, _sigma_x, total - index2 - 1)
    y = _unit_term(index1, _sigma_y, index2 - index1 - 1, _sigma_y, total - index2 - 1)
    z = _unit_term(index1, _sigma_z, index2 - index1 - 1, _sigma_z, total - index2 - 1)
    return x + y + z


@torch.jit.script
def heisenberg_hamiltonian(size: tuple[int, int], J: float) -> torch.Tensor:
    """
    Generate Hamiltonian of the Heisenberg model.
    For testing, it is recommanded to set J=1.
    This function does not work for large systems, so please do not set size larger than 16.

    Parameters
    ----------
    size : tuple[int, int]
        The size of the lattice.
    J : float
        The parameter of Heisenberg model.

    Returns
    -------
    torch.Tensor
        The Hamiltonian in the format of dense torch tensor.
    """

    L1, L2 = size
    n_qubits = L1 * L2
    assert n_qubits <= 16
    n_dim = 2**n_qubits

    result: torch.Tensor | None = None
    for l1 in range(L1):
        for l2 in range(L2):
            if l1 != 0:
                term = _single_term(_get_index((l1 - 1, l2), size), _get_index((l1, l2), size), n_qubits)
                if result is None:
                    result = term
                else:
                    result = result + term
            if l2 != 0:
                term = _single_term(_get_index((l1, l2 - 1), size), _get_index((l1, l2), size), n_qubits)
                if result is None:
                    result = term
                else:
                    result = result + term

    assert result is not None
    return result * (J / 4)
