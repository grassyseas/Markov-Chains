def dot(u, v):
    """
    Returns the dot product of two vectors u, v.
    The vectors u, v are lists.
    """
    if len(u) != len(v):
        print ("Vector Dot Product Dimension Error:", len(u), ",", len(v))
    else:
        return sum([u[i] * v[i] for i in range(len(u))])

def scale(r, M):
    """
    Scales each entry in M by the scalar r.
    """
    return [[r * M[p][q] for q in range(len(M[0]))] for p in range(len(M))]

def add(M1, M2):
    """
    Returns a matrix Q, where Q[i][j] = M1[i][j] + M2[i][j].
    M2 is replaced by Q.
    """
    m = len(M1)
    n = len(M1[0])
    for p in range(m):
        for q in range(n):
            M2[p][q] = M2[p][q] + M1[p][q]
    return M2

def mult(M1, M2):
    """
    Returns the product of two matrices M1, M2.
    The matrices M1, M2 are arrays, or lists of lists.
    """
    # M1 is a r1 x c1 matrix, and M2 is a r2 x c2 matrix.
    # If c1 = r2, then the multiplication is well-defined.
    r1, c1, r2, c2 = len(M1), len(M1[0]), len(M2), len(M2[0])
    if c1 != r2:
        print ("Matrix Multiplication Dimension Error:", c1, ",", r2)
        return
    # X will be the r1 x c2 output matrix.
    X = [[0 for q in range(c2)] for p in range(r1)]
    # Entry X(i,j) is the dot product of row i (M1) and column j (M2).
    for i in range(r1):
        for j in range(c2):
            X[i][j] = dot(M1[i], [M2[k][j] for k in range(r2)])
    return X

def mult2(M1, M2):
    """
    Multiplies two arrays M1 and M2 entry by entry.
    mult2(M1, M2)[i][j] = M1[i][j] * M2[i][j].
    """
    return [[M1[i][j] * M2[i][j] for j in range(len(M1))] for i in range(len(M2))]


def power(M, k):
    """
    Given a matrix M, it returns M^k.
    If M is a transition matrix, then the (i,j)-th entry of M^k is
    the probability that state i will reach state j in exactly k steps.
    If state j is an absorbing state, then the (i,j)-th entry is the
    probability that state i will reach state j within k steps.
    """
    Mnew = [M[x] for x in range(len(M))]
    for i in range(k):
        Mnew = mult(M, Mnew)
    return Mnew

def rinse(M, d=1e-14):
    """
    This function removes any entries in a list or array that 
    are closer to zero than d is. It is used in the gauss function
    to avoid division by near-zero values. By default d is set
    to be equal to 10^-14.
    """
    m = len(M)
    if type(M[0]) == list:
        n = len(M[0])
    else:
        for ind, x in enumerate(M):
            if abs(x) < d:
                M[ind] = 0
        return M
    for p in range(m):
        for q in range(n):
            if abs(M[p][q]) < d:
                M[p][q] = 0
    return M

def pivotindex(u):
    """
    Returns the index of the first nonzero entry in a vector u.
    """
    i = 0
    for x in u:
        if x != 0:
            break
        i = i + 1
    return i

def Rpivot(p, q, Mb):
    """
    Given an augmented matrix Mb, Mb = M|b, this gives the output of the 
    pivot entry [i, j] in or below row p, and in or to the right of column q.
    """
    # n is the number of columns of M, which is one less than that of Mb.
    m = len(Mb)
    n = len(Mb[0]) - 1
    # Initialize i, j to p, q, and we will not go above or leftwards of p, q.
    i = p
    j = q
    # Iterate through the columns of Mb to find its first nonzero column.
    for y in range(q, n):
        if [Mb[x][y] for x in range(p, m)] == [0] * (m - p):
            j = j + 1
        else:
            break
    # Iterate through the rows of M from p to n-1.
    for x in range(p, n):
        # Adds one to row index i if column i is all zeros from column j
        # to column n.
        if Mb[x][j:n] == [0] * (n - j + 1):
            i = i + 1
        else:
            break
    return [i, j]
            
def elim(i, M):
    """
    Uses row i to eliminate corresponding pivot column in array M.
    Returns result.
    """
    m = len(M)
    n = len(M[0])
    # j is the column index of the pivot in row i.
    j = pivotindex(M[i])
    
    # This condition prevents out of bounds errors.
    if j == n:
        return M
    else:
        C = M[i][j]
    # Row i is multiplied by M[x][j] / C and subtracted from row x.
    for x in range(m):
        if x != i:
            M[x] = [M[x][q] - M[i][q] * M[x][j] / C for q in range(n)]
    return M

def gauss(M, b, d=1e-14):
    """
    Solves Mx = b and returns the augmented matrix M|b in reduced row
    echelon form. M is list of lists, and b is a list.
    """
    # M is an m x n matrix.
    m = len(M)
    n = len(M[0])
    # Here M and b are combined into an augmented matrix, M = M|b.
    M = [M[q] + [b[q]] for q in range(m)]
    # Initiate current column of focus.
    q = 0
    # Iterate through the rows of M|b.
    for p in range(m):
        M, b = rinse(M, d), rinse(b, d)
        # i is the pivot row, and j is the pivot column.
        i, j = Rpivot(p, q, M)
        # Move column of next focus one to the right of the pivot column.
        q = q + 1
        # Swap pivot row i to row of focus p.
        M[p], M[i] = M[i], M[p]
        # Use row p to eliminate all nonzero terms in column j.
        M = elim(p, M)
    # Iterate through the rows of M|b.
    for x in range(m):
        # y is the column index of the first nonzero entry in row x.
        y = pivotindex(M[x])
        # If M|b has a nonzero entry in row x, excluding the entry in
        # column b, then normalize the row so that its pivot entry is one.
        if y <= n:
            M[x] = [M[x][s] / M[x][y] for s in range(n+1)]
        M, b = rinse(M, d), rinse(b, d)
    return M

def meanTime(T, d=1e-14):
    """
    The input T should be an n x n transition matrix. The output will give
    the mean hitting time of each state to the state associated with the 
    bottom row.
    """
    m = len(T)
    n = len(T[0])
    # Removes the last row and last column of T.
    T = [T[x][:n-1] for x in range(m - 1)]
    # Replaces T with T - I, where I is the identity matrix.
    for x in range(m-1):
        T[x][x] = T[x][x] - 1
    # b is a column vector of -1's.
    b = [-1] * (m - 1)
    # The matrix T|b in row reduced echelon form is returned.
    return gauss(T, b, d)

def hpTrans(Dvals, Dprobs, Rvals, Rprobs, H, printHPvals="off"):
    """
    This function transforms information about a video game battle
    into a transition matrix output. In the battle the aggressor
    randomly deals different damage values to the character, whom
    randomly heals different amounts. This continues in a back and
    forth manner, where the character stays at zero hp once they
    reach it and cannot exceed their maximum hp named H.
    Dvals is the list of possible damage values that can be inflicted.
    Dprobs is the corresponding list of probabilities that each damage value
    has of being inflicted per roll.
    Rvals is the list of possible recovery amounts per recovery round.
    Rprobs is the corresponding list of probabilities that each recovery value
    has of being recovered per roll.
    H is the starting health value, and the maximum health value.
    This function takes in the following inputs and generates from them
    a transition matrix. It does this by finding all possible health values
    that can emerge, and those health values become the possible states
    for the transition matrix. It then finds the probability of going from
    any health value to any other health value after a single round of
    taking damage and healing.
    If health would become negative, then it is set to zero instead.
    If health becomes zero, then it stays at zero permanently.
    Health cannot exceed H. If this condition is eliminated then 
    the transition matrix would typically become infinite in size and
    as a result the algorithms outlined here would not be effective.
    For dealing with such a situation, refer to writings on markov chains
    with countably infinitely many states.
    """
    
    # HPvals will become a list of all the possible HP values.
    HPvals = []
    # newHPvals will be the list of possible HP values generated
    # one step ahead of HPvals, so that when HPvals and newHPvals
    # become the same at the end of a generation cycle we will know that
    # all possible HP values have been discovered.
    newHPvals = [H, 0]
    # betweenHP is the HP after taking damage, but before recovering.
    # The purpose is that if betweenHP is 0 or less then regardless
    # of the subsequent recovery amount, the resultant HP value should be zero.
    betweenHP = 0
    
    # This process repeats until HPvals and newHPvals become the same.
    while HPvals != newHPvals:
        # First we make HPvals a copy of newHPvals.
        HPvals = [x for x in newHPvals]
        for dmg in Dvals:
            for recov in Rvals:
                for hp in HPvals:
                    betweenHP = hp - dmg
                    # newHP is the HP value after damage and recovery,
                    # but assuming betweenHP was positive.
                    newHP = hp - dmg + recov
                    # We add newHP as a new HP value if it's less than
                    # H and betweenHP was positive.
                    if betweenHP > 0 and newHP < H:
                        newHPvals.append(newHP)
        # The subsequent four lines eliminate duplicates from
        # HPvals and newHPvals and sort them so that they can be checked
        # for being identical in the next potential repeat of the process.
        newHPvals = list(dict.fromkeys(newHPvals))
        newHPvals.sort()
        HPvals = list(dict.fromkeys(HPvals))
        HPvals.sort()
    # Now that all health values have been found, we sort them from
    # greatest to least. This puts the state of having zero health
    # as the last state. With a matrix constructed in this way we can
    # input it to the meanTime algorithm and receive the mean times it takes
    # each health value (except zero) to reach zero health.
    HPvals.sort()
    HPvals = HPvals[::-1]
    if printHPvals == "on":
        print ("Possible HP values:")
        print (HPvals)
    
    # Next we construct the transition matrix for our HP values.
    m = len(HPvals)
    # M is an mxm matrix of zeroes, which will be added to to form the
    # transition matrix until it is completely formed.
    M = [[0 for i in range(m)] for i in range(m)]
    for p, hp in enumerate(HPvals):
        for k1, p1, dmg in zip(range(len(Dvals)), Dprobs, Dvals):
            for k2, p2, recov in zip(range(len(Rvals)), Rprobs, Rvals):
                betweenHP = hp - dmg
                newHP = 0
                # if betweenHP is non-positive, then the next state is
                # the zero health state.
                if betweenHP <= 0:
                    newHP = 0
                # Otherwise, the next state is the previous state with
                # damage subtracted and recovery added, but no greater than
                # H.
                else:
                    newHP = min(betweenHP + recov, H)
                # This finds the column index for the new state.
                q = HPvals.index(newHP)
                # This adds the probability of going from state p to
                # state q, and adds it to the transition matrix in
                # the corresponding position.
                M[p][q] = M[p][q] + p1 * p2
    return (M)

def meanTime2(M, i, j, N, error="off"):
    """
    This algorithm performs the same function as meanTime, but works
    differently. It takes as input the transition matrix M, and
    a beginning state i and target state j. It will estimate the average
    time it takes for state i to reach state j. It calculates this estimate
    by using powers of the transition matrix rather than by solving a system
    of linear equations. Since an infinite number of powers cannot be 
    calculated, the result is an estimate which is better the larger N is.
    N is the greatest power to calculate. If it is not large enough, then the
    result could be quite inaccurate.
    
    Advice on approaching error:
    Suppose you choose N = 1000, then if you set error to "on" it will print
    how much the the powers from 900 to 1000 contributed to the total estimate.
    The smaller this value, the better. With large enough N, an estimate 
    practically identical to the correct value is possible, but ultimately
    it depends on the nature of (M, i, j) how costly the calculation will
    be. Enabling error to "on" is a good idea for larger transition matrices.
    When the printed "Contribution of Last 10% of Powers" is small, then
    that typically indicates better accuracy for the estimate.
    Example: With N = 1000 with one (M, i, j) triplet, I got a last 10%
    contribution of 6.67 (powers 900 to 1000). By increasing N to 5000 the
    result (powers 4000 to 5000) was 0.00001656. In other words, the powers
    from 4000 to 5000 contributed little to the final mean time value, which
    indicates greater accuracy short of an unsuitable (M, i, j) triplet.
    
    Notes on the last 10% value:
    A small last 10% value does not by itself indicate an accurate estimate,
    even if it is 10^10^-100 and N is 10^10^100. Greater certainty requires
    additional mathematical work, or at least an assessment of the underlying
    markov chain.
    """
    # It takes zero steps on average for a state to reach itself.
    if i == j:
        return 0
    m = len(M)
    # total will become an estimate of the mean time for state i to reach
    # state j. It starts at 0 and then has the probability to reach
    # state j in one step, multiplied by 1 (the number of steps). Next we add
    # the probability for state i to reach state j in 2 steps, multiplied by
    # 2 (the number of steps). In this way total is monotonically increasing
    # towards the mean time.
    total = 0
    # N9 and last10 are used when error = "on".
    N9 = N * 0.9
    last10 = 0
    # X is a copy of M.
    X = [[M[p][q] for q in range(m)] for p in range(m)]
    # prev is the (i,j)-th entry of the previous power of M.
    prev = 0
    # On the k-th step we add the mean time contribution of step k to total.
    for k in range(1, N+1):
        # We replace total with itself added to the probability of reaching
        # state j on step k, minus the probability of having reached it on
        # state k-1, and all multiplied (weighted) by k.
        total = total + k * (X[i][j] - prev)
        if k >= N9 and error == "on":
            last10 = last10 + k * (X[i][j] - prev)
        # prev becomes M^k[i][j]
        prev = X[i][j]
        # X becomes M^(k+1)
        X = mult(M, X)
    if error == "on":
        print ("Contribution of Last 10% of Powers:", last10)
    return total

# Guide to Using Functions
"""
First generate a transition matrix. A transition matrix is a square matrix with
probabilities for entries which sum to 1 for each row. A transition matrix
can either be created manually or by defining Dvals, Dprobs, Rvals, Rprobs, and
H, and feeding them into hpTrans which will convert them to a transition
matrix. Next you can take powers of a transition matrix with the power function
in order to see the probability for a state i to reach a state j in exactly
k steps (if the power taken of the transition matrix is k.) It's possible to
calculate the mean time for states to reach another state as well. With
meanTime you can input a transition matrix and it will solve the mean time
for each state to reach the state in the bottom row. If you don't want to
find the mean time to reach the state in the bottom row of the transition
matrix, then swap the bottom row with the row associated with the state 
that you are interested in. This can be done manually or with swap.
meanTime2 takes your transition matrix as input as well as a starting stage
and target state which it will find the mean time for. Be careful to enable
error to "on" for larger transition matrices and exercise judgment on
how inaccurate the results might be. The function gauss can be used to solve
a system of linear equations. It takes a matrix input and a vector b, and solves
Mx = b by forming an augmented matrix M|b. The functions pivotindex, swap,
Rpivot, elim, and rinse are mainly there to be used by the gauss function.
The function dot is used to define mult, so that matrix multiplication is
possible. The function mult is used to define power. The functions add, scale,
and mult2 are not used, but might have further applications which is why
they are included.
"""

# Example

# First we define the inputs for hpTrans.
Dvals = [80, 60, 40, 20, 0]
Dprobs = [1/5, 1/5, 1/5, 1/5, 1/5]
Rvals = [100, 80, 60, 40, 0]
Rprobs = [1/5, 1/5, 1/5, 1/5, 1/5]
H = 100

# From these we get our transition matrix.
T = hpTrans(Dvals, Dprobs, Rvals, Rprobs, H, printHPvals="on")
print ("Transition Matrix:")
print (T)
print ("")
#  We can take powers of this matrix.
T10 = power(T, 10)
print ("10th Power of Transition Matrix:")
print (T10)
print ("")

# We can store the row / column count of T as m.
m = len(T)
# We can find the average number of steps from each possible health value
# to 0 hp.
avgs = meanTime(T)

# The avgs are the last number in each list.
# The average steps for state 0 / 100 hp:
print ("Average steps from 100 hp to 0 hp:")
print (avgs[0][-1])
# The average steps for state 2 / 60 hp:
print ("Average steps from 60 hp to 0 hp:")
print (avgs[2][-1])
print ("")
# We can find the average steps from 200 hp to 0 hp using meanTime2.
# The last state is the length of T minus 1.
avg = meanTime2(T, 0, m - 1, 10, error="on")
print ("N=10")
print ("Estimated average steps from state 100 hp to 0 hp:")
print (avg)
print ("")
# The contribution of the last 10% of powers was high, so we increase N.
print ("N=500")
avg2 = meanTime2(T, 0, m - 1, 500, error="on")
print ("Estimated average steps from state 100 hp to 0 hp:")
print (avg2)
# This time the result is very accurate, and matches that from meanTime.
