from Types import Matrix as M

A = M(3, 2)
B = M(2, 3)

A.data = [[5,4],[7,6],[9,8]]
B.data = [[9,8,5],[7,6,4]]

E = A.product(B)

print(f'A.product(B): {E.data}')

print(f'A Original:{A.data}')
A.T()
print(f'A Transpose:{A.data}, A.Shape:{A.shape()}')
A.T()
print(f'A Original: {A.data}, A.Shape: {A.shape()}')
A.reshape(3, 3)
print(f'A Resized: {A.data}, A.Shape {A.shape()}')

print(f'B Original: {B.data}')
D = B.copy()
print(f'D Copy of B: {D.data}')
B.reshape(3, 3)
print(f'B Resized: {B.data}')

C = A.product(B)

print(f'A.Product(B): {C.data}')

T = M(3, 3)

T.data = [[2, 3, 1],
          [4, 5, 6],
          [7, 8, 9]]

print(f'Rank of T: {T.rank()}')

B = T.invert()

D = T.det()

print(f'Inverse of T: {B.data}')
print(f'Determinant of T: {D}')

U = M(3, 3)

U.data = [[1, 4, 7],
          [3, 0, 5],
          [-1, 9, 11]]

subU = U.submatrix(2, 3)
minor = U.minor(2, 3)
coF = U.cofactor(2, 3)

print(f'U: {U.data}')
print(f'Sub 2,3 of U: {subU.data}')
print(f'Minor 2,3 of U: {minor}')
print(f'Cofactor 2,3 of U: {coF}')