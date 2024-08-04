import torch
torch.set_printoptions(precision=2, sci_mode=False)

embeddings = torch.tensor(
    [[1, 0, 0],  # boy
     [0, 1, 0],  # bites
     [0, 0, 1]], dtype=torch.float)  # dog

K_W = torch.tensor(
    [[0.22, .3, .2],
     [.4, 3, -.4],
     [-.2, 1, .33]])

Q_W = torch.tensor(
    [[-0.23, .13, -.7],
     [.4, 3, -.46],
     [-.22, 1, .36]])

V_W = torch.tensor(
    [[1.0, 0, 0],
     [0, 1.0, 0],
     [0, 0, 1.0]])

O_W = torch.tensor(
    [[.1, 0, 0],
     [0, .1, 0],
     [0, 0, .1]])

K = embeddings.mm(K_W)
Q = embeddings.mm(Q_W)
V = embeddings.mm(V_W)

M = Q.mm(K.t())
S = M.softmax(dim=1)

A = S.mm(V)

Delta = A.mm(O_W)

New = embeddings + Delta

print("Weights:")
print(f"Q_W:\n{Q_W}")
print(f"K_W:\n{K_W}")
print(f"V_W:\n{V_W}")
print(f"O_W:\n{O_W}")

print("Computation:")
print(f"Input Embeddings:\n{embeddings}")
print(f"Q:\n{Q}")
print(f"K:\n{K}")
print(f"K.T:\n{K.T}")
print(f"V:\n{V}")
print(f"M:\n{M}")
print(f"S:\n{S}")
print(f"A:\n{A}")
print(f"Delta:\n{Delta}")
print(f"Output Embeddings:\n{New}")
print(f"Normalized Output Embeddings :\n{New/torch.sum(New, axis=1, keepdim=True)}")
