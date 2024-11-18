import torch

import torch

# Tworzenie przykładowych macierzy
A = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
B = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)

# Dodawanie macierzy
C = A + B
print("Dodawanie macierzy A + B:\n", C)

# Mnożenie elementów macierzy (element-wise)
D = A * B
print("Mnożenie elementów macierzy A * B:\n", D)

# Mnożenie macierzy (produkt macierzy)
E = torch.mm(A, B.T)  # Transponujemy B, aby wymiary pasowały do mnożenia
print("Mnożenie macierzy A * B^T:\n", E)

# Transpozycja macierzy
F = A.T
print("Transpozycja macierzy A:\n", F)