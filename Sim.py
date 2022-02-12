from Types import Matrix as M
from decimal import *

A = M(3, 2)
B = M(2, 3)

A.data = [[5,4],[7,6],[9,8]]
B.data = [[9,8,5],[7,6,4]]

E = A.product(B)

print(f'A.product(B):\n{E}')

print(f'A Original:\n{A}')
A.T()
print(f'A Transpose:\n{A}, A.Shape:{A.shape()}')
A.T()
print(f'A Original:\n{A}, A.Shape: {A.shape()}')
A.reshape(3, 3)
print(f'A Resized:\n{A}, A.Shape {A.shape()}')

print(f'B Original:\n{B}')
D = B.copy()
print(f'D Copy of B:\n{D}')
B.reshape(3, 3)
print(f'B Resized:\n{B}')

C = A.product(B)

print(f'A.Product(B):\n{C}')

T = M(3, 3)

T.data = [[2, 3, 1],
          [4, 5, 6],
          [7, 8, 9]]

print(f'Rank of T: {T.rank()}')

B = T.invert()

B.round(10)

D = T.det()

print(f'Inverse of T:\n{B}')
print(f'Determinant of T: {D}')

U = M(3, 3)

U.data = [[12, -51, 4],
          [6, 167, -68],
          [-4, 24, -41]]

(Q, R) = U.QR()

print(f'U: \n{U}')
print(f'Q: \n{Q}')
print(f'R: \n{R}')