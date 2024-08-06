## Code taken from the original InterFaceGAN Repo here: https://github.com/genforce/interfacegan/blob/8da3fc0fe2a1d4c88dc5f9bee65e8077093ad2bb/utils/manipulator.py#L154

import numpy as np

def project_boundary(primal, *args):
  """Projects the primal boundary onto condition boundaries.
  
  The function is used for conditional manipulation, where the projected vector
  will be subscribed from the normal direction of the original boundary. Here,
  all input boundaries are supposed to have already been normalized to unit
  norm, and with same shape [1, latent_space_dim].
  
  Args:
    primal: The primal boundary.
    *args: Other boundaries as conditions.
  
  Returns:
    A projected boundary (also normalized to unit norm), which is orthogonal to
      all condition boundaries.
  
  Raises:
    LinAlgError: If there are more than two condition boundaries and the method fails 
                 to find a projected boundary orthogonal to all condition boundaries.
  """
  assert len(primal.shape) == 2 and primal.shape[0] == 1

  if not args:
    return primal
  if len(args) == 1:
    cond = args[0]
    assert (len(cond.shape) == 2 and cond.shape[0] == 1 and
            cond.shape[1] == primal.shape[1])
    new = primal - primal.dot(cond.T) * cond
    return new / np.linalg.norm(new)
  elif len(args) == 2:
    cond_1 = args[0]
    cond_2 = args[1]
    assert (len(cond_1.shape) == 2 and cond_1.shape[0] == 1 and
            cond_1.shape[1] == primal.shape[1])
    assert (len(cond_2.shape) == 2 and cond_2.shape[0] == 1 and
            cond_2.shape[1] == primal.shape[1])
    primal_cond_1 = primal.dot(cond_1.T)
    primal_cond_2 = primal.dot(cond_2.T)
    cond_1_cond_2 = cond_1.dot(cond_2.T)
    alpha = (primal_cond_1 - primal_cond_2 * cond_1_cond_2) / (
        1 - cond_1_cond_2 ** 2 + 1e-8)
    beta = (primal_cond_2 - primal_cond_1 * cond_1_cond_2) / (
        1 - cond_1_cond_2 ** 2 + 1e-8)
    new = primal - alpha * cond_1 - beta * cond_2
    return new / np.linalg.norm(new)
  else:
    for cond_boundary in args:
      assert (len(cond_boundary.shape) == 2 and cond_boundary.shape[0] == 1 and
              cond_boundary.shape[1] == primal.shape[1])
    cond_boundaries = np.squeeze(np.asarray(args))
    A = np.matmul(cond_boundaries, cond_boundaries.T)
    B = np.matmul(cond_boundaries, primal.T)
    x = np.linalg.solve(A, B)
    new = primal - (np.matmul(x.T, cond_boundaries))
    return new / np.linalg.norm(new)
