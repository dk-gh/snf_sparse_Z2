import numpy as np
import statistics


class DesynchronisationError(Exception):
    pass


class ShapeError(Exception):
    pass


class SparseBinaryMatrix:

    def __init__(self):

        # rows consist of a list of sets of ints
        self.rows = []
        # columns consists of a list of sets of ints
        self.columns = []

        self.shape = (0, 0)

        # flag set indicating whether matrix is
        # currently in smith normal form
        self.is_smith_normal_form = False

    @staticmethod
    def zeros(n, m):
        # creates an n*m sparse matrix initialised to all zeros
        # equivalent dense numpy array would be np.zeros(n*m).reshape(n,m)
        sm = SparseBinaryMatrix()
        sm.shape = (n, m)
        sm.rows = [set() for _ in range(n)]
        sm.columns = [set() for _ in range(m)]
        return sm

    def swap_rows(self, n, m):

        columns_to_update = self.rows[m] ^ self.rows[n]

        for column in columns_to_update:
            if n in self.columns[column]:
                self.columns[column].discard(n)
            else:
                self.columns[column].add(n)

            if m in self.columns[column]:
                self.columns[column].discard(m)
            else:
                self.columns[column].add(m)

        self.rows[n], self.rows[m] = self.rows[m], self.rows[n]

    def swap_columns(self, n, m):

        rows_to_update = self.columns[m] ^ self.columns[n]

        for row in rows_to_update:
            if n in self.rows[row]:
                self.rows[row].discard(n)
            else:
                self.rows[row].add(n)

            if m in self.rows[row]:
                self.rows[row].discard(m)
            else:
                self.rows[row].add(m)

        self.columns[n], self.columns[m] = self.columns[m], self.columns[n]

    def xor_rows(self, n, m):

        for column in self.rows[m]:
            if n in self.columns[column]:
                if m in self.columns[column]:
                    self.columns[column].discard(n)
                else:
                    self.columns[column].add(n)
            else:
                self.columns[column].add(n)

        for column in self.rows[n] - self.rows[m]:
            if n in self.columns[column]:
                if m in self.columns[column]:
                    self.columns[column].discard(n)
                else:
                    self.columns[column].add(n)
            else:
                self.columns[column].add(n)

        self.rows[n] = self.rows[m] ^ self.rows[n]

    def xor_columns(self, n, m):

        for row in self.columns[m]:
            if n in self.rows[row]:
                if m in self.rows[row]:
                    self.rows[row].discard(n)
                else:
                    self.rows[row].add(n)
            else:
                self.rows[row].add(n)

        for row in self.columns[n] - self.columns[m]:
            if n in self.rows[row]:
                if m in self.rows[row]:
                    self.rows[row].discard(n)
                else:
                    self.rows[row].add(n)
            else:
                self.rows[row].add(n)

        self.columns[n] = self.columns[m] ^ self.columns[n]

    def nnz(self, validate=False):
        '''

        Returns
        -------
        int
        number of non-zero entries in matrix

        '''
        nnz = sum([len(x) for x in self.rows])

        if validate:
            col_nnz = sum([len(x) for x in self.columns])
            if nnz != col_nnz:
                raise DesynchronisationError()

        return nnz

    def smith_normal_form(self):

        n = 0

        while n < min(self.shape):

            for i, row in enumerate(self.rows):
                if i < n:
                    pass
                elif row:
                    for j in row:
                        if j > n:
                            break

                    self.swap_rows(n, i)
                    self.swap_columns(n, j)
                    break

            for i in self.columns[n]-{n}:
                self.xor_rows(i, n)

            for j in self.rows[n]-{n}:
                self.xor_columns(j, n)
            n = n+1

        self.is_smith_normal_form = True

    def smith_normal_form_profiled(self):
        """
        This is an identical implementation of the smith_normal_form method
        but at each step we track how the sparsity changes over the course of
        the algorithm.

        This allows us to see, for example, the space requirements of
        the algorithm.

        At each step we record:

            the total number of non-zero entries in the matrix -- nnz
            the largest number of non-zero entries in any row -- r_max
            the largest number of non-zero entries in any column -- c_max
            the average number of non-zero entries in rows -- r_mean
            the average number of non-zero entries in columns -- c_mean

        Each of these values are stored in a list, with new values appended
        at each step of the algorithm.

        The argument returns these lists for performance analysis.

        Returns
        -------
        nnz_list : list

        r_max_list : list

        c_max_list : list

        r_mean_list : list

        c_mean_list : list

        """
        n = 0
        nnz_list = []
        r_max_list = []
        c_max_list = []
        r_mean_list = []
        c_mean_list = []
        while n < min(self.shape):
            nnz_list.append(self.nnz())
            r_lens = [len(x) for x in self.rows]
            c_lens = [len(x) for x in self.columns]
            r_max_list.append(max(r_lens))
            c_max_list.append(max(c_lens))

            r_mean_list.append(statistics.mean(r_lens))
            c_mean_list.append(statistics.mean(c_lens))

            for i, row in enumerate(self.rows):
                if i < n:
                    pass
                elif row:
                    for j in row:
                        if j > n:
                            break

                    self.swap_rows(n, i)
                    self.swap_columns(n, j)
                    break

            for i in self.columns[n]-{n}:
                self.xor_rows(i, n)

            for j in self.rows[n]-{n}:
                self.xor_columns(j, n)
            n = n+1

        self.is_smith_normal_form = True
        return nnz_list, r_max_list, c_max_list, r_mean_list, c_mean_list

    def lookup(self, i, j):
        """
        lookup a specific value in the matrix

        Parameters
        ----------
        i : int
            i is a row index
        j : int
            j is a column index

        Returns
        -------
        int
            the value of the matrix at [i,j]
        """

        if j in self.rows[i]:
            return 1
        else:
            return 0

    def trace(self):
        """
        returns the trace of the matrix

        Returns
        -------
        tr : int
            the trace of the matrix

        """

        # if the matrix is already in smith normal form
        # then computing trace is merely counting the rows
        # which are not empty
        if self.is_smith_normal_form:
            return self.nnz()

        # if the matrix is not in smith normal form then
        # we must sum over the full diagonal
        else:
            tr = 0
            for i in range(np.min(self.shape)):
                tr += self.lookup(i, i)
            return tr

    def validate_synchronisation(self):
        """
        This method checks that both row and column-based
        representations are synchronised, i.e. both represent
        the same underlying matrix.

        This method is generally used in testing during refactoring.

        Raises
        ------
        ShapeError
            DESCRIPTION.
        DesynchronisationError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.shape[0] != len(self.rows):
            raise ShapeError()

        if self.shape[1] != len(self.columns):
            raise ShapeError()

        for row_index, row in enumerate(self.rows):
            for column_index in row:
                if row_index not in self.columns[column_index]:
                    raise DesynchronisationError()

        for column_index, column in enumerate(self.columns):
            for row_index in column:
                if column_index not in self.rows[row_index]:
                    raise DesynchronisationError()

    def from_numpy_array(self, array, validate=False):
        """
        Creates a sparse matrix from a numpy array.

        Parameters
        ----------
        array : numpy array
            numpy array with int entries 0,1
        validate : TYPE, optional
            If validate=True then the validate_synchronisation
            method is called. The default is False.

        Returns
        -------
        None.

        """

        self.shape = array.shape

        for array_row in array:
            row_set = set()
            for i, value in enumerate(array_row):
                if value == 1:
                    row_set.add(i)
            self.rows.append(row_set)

        for array_column in array.T:
            column_set = set()
            for i, value in enumerate(array_column):
                if value == 1:
                    column_set.add(i)
            self.columns.append(column_set)

        if validate:
            self.validate_synchronisation()

    def to_numpy_array(self, validate=False):
        """
        Outputs the sparse matrix as a numpy array. Uses the row data
        to create the matrix entries.

        Parameters
        ----------
        validate : TYPE, optional
            DESCRIPTION. The default is False.

        Raises
        ------
        DesynchronisationError
            If validate=True then:
                1) the validate_synchronisation method is called
                2) a second numpy array is created using the column
                   data as well and checks that both numpy arrays
                   are equal.

        Returns
        -------
        array : numpy array
            The matrix is returned as a dense matrix in the form
            of a numpy array.

        """

        array = np.zeros(self.shape, dtype=int)

        for i, row in enumerate(self.rows):
            for value in row:
                array[i, value] = 1

        if validate:
            self.validate_synchronisation()

            validation_array = np.zeros(self.shape, dtype=int).T
            for i, column in enumerate(self.columns):
                for value in column:
                    validation_array[i, value] = 1
            validation_array = validation_array.T

            if not (array == validation_array).all():
                raise DesynchronisationError()

        return array


if __name__ == '__main__':

    X = np.array(
        [[0, 1, 1, 0, 1],
         [1, 0, 0, 1, 0],
         [1, 0, 1, 0, 1],
         [1, 1, 0, 1, 1]]
    )

    m = SparseBinaryMatrix()
    m.from_numpy_array(X)
    print('input matrix:\n')
    print(f'{X}\n')
    print(f'row representation: {m.rows}')
    print(f'column representation: {m.columns}\n')

    print('-'*45)

    print('swapping rows 0 and 2:')
    m.swap_rows(0, 2)
    X = m.to_numpy_array()

    print('resulting matrix:\n')
    print(f'{X}\n')
    print(f'row representation: {m.rows}')
    print(f'column representation: {m.columns}\n')

    print('-'*45)

    print('swapping columns 0 and 1:')
    m.swap_columns(0, 1)
    X = m.to_numpy_array()

    print('resulting matrix:\n')
    print(f'{X}\n')
    print(f'row representation: {m.rows}')
    print(f'column representation: {m.columns}\n')

    print('-'*45)

    print('xor columns 0 and 1')
    m.xor_columns(0, 1)
    X = m.to_numpy_array()

    print('resulting matrix:\n')
    print(f'{X}\n')
    print(f'row representation: {m.rows}')
    print(f'column representation: {m.columns}\n')

    print('-'*45)

    print('compute smith normal form')
    m.smith_normal_form()
    X = m.to_numpy_array()

    print('resulting matrix:\n')
    print(f'{X}\n')
    print(f'row representation: {m.rows}')
    print(f'column representation: {m.columns}\n')

    print('-'*45)

    print('initialise a 2x5 matrix with all zero entries')
    m = SparseBinaryMatrix.zeros(2, 5)
    X = m.to_numpy_array()

    print('resulting matrix:\n')
    print(f'{X}\n')
    print(f'row representation: {m.rows}')
    print(f'column representation: {m.columns}\n')
