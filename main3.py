__author__ = 'tomringstrom'
"""
Array Backed Grid

Show how to use a two-dimensional list/array to back the display of a
grid on-screen.

If Python and Arcade are installed, this example can be run from the command line with:
python -m arcade.examples.array_backed_grid

Some nomenclature:
LI: linear index
Terminal: Goal State
"""
# TODO:  In the policy optimization, there are often times where the agent can select any policy within an acceptable
# TODO:  set, often this selection doesn't reduce the total time taken since it is determined by p(sat) which often = 1.
# One of the things that might mess up the algorithm is that the max ent distribution is too entropic, which gives
# the agent a small amount of confidence that something impossible is possible (i.e. moving 8 steps into the future when
# we condition on 5 time steps.  The controller simply won't allow this.

# TODO: The problem with state-value equivalence for p(reach) could just be solved by adding a small significant heavy
# TODO: tail for the entire deadline distribution (but only for the goal deadline)

import arcade
import timeit
import torch
import itertools
import numpy as np
import torch.nn.functional as f
import pickle
import os.path
import scipy as sp
from scipy.sparse import csr_matrix as csr
from scipy.stats import norm
from scipy.linalg import hankel
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import spsolve
import signal, sys, os
from collections import defaultdict
from itertools import chain, combinations, permutations
import ggplot as gg

def sigquit_handler(signum, frame):
    print('SIGQUIT received; exiting')
    sys.exit(os.EX_SOFTWARE)
signal.signal(signal.SIGQUIT, sigquit_handler)


HALF_SQUARE_WIDTH = 59
SQUARE_WIDTH = HALF_SQUARE_WIDTH * 2
HALF_SQUARE_HEIGHT = 59
SQUARE_HEIGHT = HALF_SQUARE_HEIGHT * 2
SPACER = 1
SQUARE_SPACING = int(SQUARE_WIDTH + SPACER)
TOTAL_WIDTH_SPACING = int(HALF_SQUARE_WIDTH*2+(SQUARE_SPACING//2))
TOTAL_HEIGHT_SPACING = int(HALF_SQUARE_HEIGHT*2+(SQUARE_SPACING//2))

# Set how many rows and columns we will have
# ROW_COUNT = 5
# COLUMN_COUNT = 13
ROW_COUNT = 5
COLUMN_COUNT = 5

SCREEN_WIDTH = SQUARE_SPACING*COLUMN_COUNT-SPACER
SCREEN_HEIGHT = SQUARE_SPACING*ROW_COUNT-SPACER

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 3
HEIGHT = 3

# This sets the margin between each cell
# and on the edges of the screen.
MARGIN = 1

MOVEMENT_SPEED = 1
MAX_SPEED = 1
MIN_SPEED = -1
STARTX = 2
STARTY = 2

# Do the math to figure out screen dimensions
# SCREEN_WIDTH = (WIDTH + MARGIN) * COLUMN_COUNT + MARGIN
# SCREEN_HEIGHT = (HEIGHT + MARGIN) * ROW_COUNT + MARGIN

# Some useful diagnostic tools
# rs = self.bigTTGmat[:,5].reshape(self.ns, self.ns)

class Agent:
    def __init__(self, x, y, color, wmat, *ss):
        """ Initialize our rectangle variables """
        self.wmat = wmat
        self.currentSS = ss
        # Position
        self.x = x
        self.y = y

        self.successflag = False

        self.timePeriodList = None
        self.solution = None

        self.currentTime = 0
        self.currentPolicy = 0
        self.worldIdx = 'main'

        # Vector
        self.delta_x = 0
        self.delta_y = 0

        # Allowable States
        self.xRange = [i for i in range(COLUMN_COUNT)]
        self.yRange = [i for i in range(ROW_COUNT)]
        self.dxRange = [i for i in range(MIN_SPEED, MAX_SPEED + 1)]
        self.dyRange = [i for i in range(MIN_SPEED, MAX_SPEED + 1)]

        self.crossDxDy = list(itertools.product(self.dxRange, self.dyRange))
        self.DyDx = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]])

        self.coordSet = list(itertools.product(self.yRange, self.xRange))
        self.coordDict = dict(zip(self.coordSet, [i for i in range(len(self.coordSet))]))

        # Color
        self.color = color

        self.plan = None
        self.planIndex = 0
        self.currentGoal = None
        self.stichSetForPlan = None
        self.planStates = []
        self.completedGoals = []


        # piSetList is a list of piSets, which map periods to policies.
        # Each member of piSetList corresponds to a goal in goalList
        self.piSetList = None

    def initializePlan(self, goalstates, plan, stichSet, initGoal, initPol):
        self.goalStates = np.array(goalstates)
        self.plan = plan
        self.stichSetForPlan = stichSet
        self.currentGoal = initGoal
        self.planStates = self.goalStates[plan]
        self.completedGoals = []
        self.currentPolicy = initPol

    def setNextGoalAndStitch(self):
        # self.planAchieved[self.planIndex] = 1
        self.completedGoals.append(self.planStates[self.planIndex])
        # self.goalsAchieved[self.plan.index(self.planIndex)] = 1
        self.planIndex = min(self.planIndex + 1, len(self.plan)-1)  # This is set to -2 in the case that you have a padding environment on the end.  Set it to -1 if not.
        self.currentGoal = self.plan[self.planIndex]
        self.currentPolicy = self.stichSetForPlan[self.planIndex]

    def getLIss(self):
        return self.coordDict[(self.y, self.x)]

    def moveForSS(self):
        self.x = max(0, min(self.x + self.delta_x, COLUMN_COUNT - 1))
        self.y = max(0, min(self.y + self.delta_y, ROW_COUNT - 1))
        # if self.wmat[self.y, self.x] != 1:
        #     self.x = max(0, min(self.x + self.delta_x, COLUMN_COUNT - 1))
        #     self.y = max(0, min(self.y + self.delta_y, ROW_COUNT - 1))

    def moveForInput(self):
        newx = max(0, min(self.x + self.delta_x, COLUMN_COUNT - 1))
        newy = max(0, min(self.y + self.delta_y, ROW_COUNT - 1))

        if self.wmat[newy, newx] == 0:
            self.x = newx
            self.y = newy

    def moveForSolution(self, nextstatedist):
        nextstate = np.random.multinomial(1, nextstatedist.todense().tolist()[0])  # Ugly code
        linind = np.where(nextstate == 1)[0][0] % self.currentSS.ns
        yx = self.currentSS.liToCoord[linind]
        self.y = yx[0]
        self.x = yx[1]
        self.currentTime += 1

    def moveForSolutionDeterministic(self, nextstatedist):
        dense = nextstatedist.todense()
        nextstateli = np.argwhere(dense == np.max(dense))[0, 1] % self.currentSS.ns   # Ugly code
        yx = self.currentSS.liToCoord[nextstateli]
        self.y = yx[0]
        self.x = yx[1]
        self.currentTime += 1

class StateSpace:
    def __init__(self, ny, nx, wallMatrix, buildTM):
        self.ns = ny*nx
        self.actionsNames = ['U', 'D', 'L', 'R']
        self.actionNums = [i for i in range(len(self.actionsNames))]
        self.actionDict = dict(zip(self.actionsNames, self.actionNums))
        self.xRange = [i for i in range(COLUMN_COUNT)]
        self.yRange = [i for i in range(ROW_COUNT)]
        self.coordSet = list(itertools.product(self.yRange, self.xRange))
        self.coordDict = dict(zip(self.coordSet, [i for i in range(len(self.coordSet))]))
        self.liToCoord = dict(zip([i for i in range(len(self.coordSet))], self.coordSet))

        self.allLinearInds = range(len(self.coordSet))
        self.wallMatrix = wallMatrix
        self.validList = [i for (i, x) in enumerate(self.wallMatrix.flatten()) if x == 0]
        self.wallList = [i for (i, x) in enumerate(self.wallMatrix.flatten()) if x == 1]
        self.validCoords = [self.liToCoord[i] for i in self.validList]

        self.nv = len(self.validList)

        yset = set([self.coordDict[(y, x)] for y in range(ROW_COUNT) for x in [0, COLUMN_COUNT - 1]])
        xset = set([self.coordDict[(y, x)] for y in [0, ROW_COUNT - 1] for x in range(COLUMN_COUNT)])
        self.perimLI = list(set.union(xset, yset))
        self.perimCoords = [self.liToCoord[i] for i in self.perimLI]

        if buildTM:
            self.buildTransitionMatrix()

    def buildTransitionMatrix(self):
        agent = Agent(0, 0, arcade.color.RED, self.wallMatrix)
        P = torch.zeros([self.ns, self.ns], dtype=torch.double)
        for yx in self.coordSet:
            for idx, dydx in enumerate(agent.DyDx):
                oldS = agent.coordDict[(yx[0], yx[1])]
                agent.y = yx[0]
                agent.x = yx[1]
                agent.delta_y = dydx[0]
                agent.delta_x = dydx[1]
                agent.moveForSS()
                newS = agent.coordDict[(agent.y, agent.x)]
                P[oldS, newS] = 1

        P = f.normalize(P, p=1, dim=1)
        # self.sP = to_sparse(self.P)
        self.csrP = csr(P.numpy(),dtype=np.double)
        del P

    # def getwallMatrix(self):

# LMDP: Linear Markov Decision Process
class LMDP:
    def __init__(self, stateSpace, baseCost, intrinsicDLmean, intrinsicDLvar, goalDLmeans, goalDLvars, mvmntDLrefMean):
        self.ss = stateSpace
        self.zDict = {}  # Desirability Functions
        self.uDict = {}  # Policies
        self.ttgDict = {}  # Policies
        self.maxTime = self.ss.ns*2
        self.ns = self.ss.ns
        self.nv = len(self.ss.validList)  # Number of valid states
        self.nbv = self.nv * self.ns
        self.t_meanList = []

        # Set goal and deadline information
        self.bigIdxList = np.linspace(0, self.ns ** 2, self.ns + 1).astype(int)

        if os.path.isfile('savedLMDPInfo.obj'):
            filehandler = open('savedLMDPInfo.obj', 'rb')
            loadedLMDP = pickle.load(filehandler)
            self.zDict = loadedLMDP['zDict']
            self.uDict = loadedLMDP['uDict']
            # self.ttgDict = loadedLMDP['ttgDict']
            self.bigTTGmat = loadedLMDP['ttg']
            self.deadlinePMF = loadedLMDP['deadline']
            self.maxTime = 40
        else:
            # Compute Desirability Functions
            nullpassive = csr(sp.sparse.identity(self.ns))
            print("Constructing Cost Functions")
            self.cFuncAll = []
            for i in range(self.ns):
                # self.cFuncAll.append(self.createCostFunction(i, self.ss.wallList, baseCost))
                if i not in self.ss.wallList:
                    self.cFuncAll.append(self.createCostFunction(i, self.ss.wallList, baseCost))
                else:
                    self.cFuncAll.append(np.ones(self.ns))
            print("Cost Functions Complete")
            # Compute Policies
            self.bigQ = sp.sparse.diags(np.array(self.cFuncAll).flatten())
            passive_list = [self.ss.csrP]*self.ns
            # If the goal state is in a wall, we set the passive dynamics to the nullPassive (Do nothing)
            for i in self.ss.wallList:
                passive_list[i] = nullpassive
            self.bigP = csr(sp.sparse.block_diag(passive_list))
            print("Computing Desirability Functions")
            self.bigZ = self.computeZFunction(self.bigQ, self.bigP)
            print("Desirability Functions Complete")
            self.Z = np.reshape(self.bigZ, [self.ns, self.ns])
            print("Computing Policies")
            self.bigU = self.computePolicy(self.bigZ, self.bigP)
            print("Policy Computation Complete")


            # Compute time-to-go on self.bigU

            bigTerminalList = [i*self.ss.ns+j for i, j in enumerate(self.ss.validList)]
            bigTerminalListFull = [i * self.ss.ns + j for i, j in enumerate(np.arange(self.ss.ns))]
            self.bigTTGdist = self.computeTimeToGo(self.bigU, bigTerminalListFull)

            # ttgTensor should contain cdf info
            self.bigTTGmat = np.array([self.bigTTGdist.cdf(t + 0.5) for t in range(0, self.maxTime)]).transpose()
            self.bigTTGmat[self.bigTTGmat<0.98] = 0  # TODO: This is a temporary patch.  Figure out if it is correct to use the 0.5 offset above.  Is there a more formal method so that we don't have ttg probabilities less than 1?

        self.intrinsicDLmean = intrinsicDLmean
        self.intrinsicDLvar = intrinsicDLvar
        self.movementDLMean = intrinsicDLmean
        self.movementDLVar = 0.0001
        self.goalDLmeans = goalDLmeans
        self.goalDLvars = goalDLvars
        self.mvmntDLrefMean = mvmntDLrefMean

        self.goalHankels = []
        self.goalDLpmfs = []
        for i in range(len(goalDLmeans)):
            dist = sp.stats.norm(self.goalDLmeans[i], self.goalDLvars[i])
            dl_pmf = self.getDeadlinePMF(dist, addHeavyTail=True)
            goal_hankel = hankel(dl_pmf)
            self.goalDLpmfs.append(dl_pmf)
            self.goalHankels.append(goal_hankel)

        # Create Intrinsic Deadline Convolution Matrix
        self.intrDLdist = sp.stats.norm(self.intrinsicDLmean, self.intrinsicDLvar)
        self.intrDLPMF = self.getDeadlinePMF(self.intrDLdist, addHeavyTail=True)
        self.intrDLHankel = hankel(self.intrDLPMF)

        # Movement deadlines represent the agent's knowledge of how long the periods are.
        self.movementDLdist = sp.stats.norm(self.mvmntDLrefMean, 0.001)
        self.movementDLPMF = self.getDeadlinePMF(self.movementDLdist, addHeavyTail=False)
        self.movementDLHankel = hankel(self.movementDLPMF)


        # Compute convolution for the probability of successfully reaching a goal state under policy k and intrinsic deadline.
        self.successfulReachBig = np.dot(self.intrDLHankel, self.bigTTGmat.transpose())
        self.successfulReachBigMovement = np.dot(self.movementDLHankel, self.bigTTGmat.transpose())

        # u = self.getPolicy(60).todense()
        # u[u < 0.03] = 0
        #
        # yx = np.array([np.array([np.array(self.ss.liToCoord[j]) for j in row.nonzero()[1]]) for row in u])
        # yx = [[[i]+list(self.ss.liToCoord[j]) for j in row.nonzero()[1]] for i, row in enumerate(u)]
        #
        # X,Y,U,V = [],[],[],[]
        # for i in range(u.shape[0]):
        #     if i not in self.ss.wallList:
        #         for j in u[i,:].nonzero()[1]:
        #             xx = self.ss.liToCoord[i][1]
        #             yy = self.ss.liToCoord[i][0]
        #             uu = (self.ss.liToCoord[j][1] - xx) * u[i,j]
        #             vv = (self.ss.liToCoord[j][0] - yy) * u[i,j]
        #             X.append(xx)
        #             Y.append(yy)
        #             U.append(uu)
        #             V.append(vv)
        #
        # X = np.array(X) + 0.5
        # Y = np.array(Y) + 0.5
        # U = np.array(U)
        # V = np.array(V)
        #
        # plt.hold(True)
        # plt.quiver(X, Y, U, V, width=0.0037, scale=18, headwidth=3, headlength=3, headaxislength=3)
        # # plt.quiver(X, Y, U, V, scale=18)
        # plt.show()


        # for k, _ in enumerate(self.piSet):
        #     print(k)
        #     endpnts = np.array([self.cEnv.ss.liToCoord[j] for j in self.piSet[k]])
        #     xarrowlength = endpnts[:, 1] - X.flatten()
        #     yarrowlength = endpnts[:, 0] - Y.flatten()
        #     plt.figure()
        #     plt.title('Agent Dynamics, Period 0')
        #     plt.ylim(0, ROW_COUNT + 1)
        #     plt.xlim(0, COLUMN_COUNT + 1)
        #     for i, a in enumerate(Xf):
        #         plt.arrow(Xf[i], Yf[i], xarrowlength[i], yarrowlength[i], head_width=0.15, length_includes_head=False, head_length=0.1, fc='k', ec='k')
        #
        #     plt.show()
        # plt.quiver(X, Y, endpnts[:, 1], endpnts[:, 0])
        # plt.show()


        # filehandler = open('savedLMDPInfo_allgaps.obj', 'wb')
        # lmdpInfo = {'zDict': self.zDict, 'uDict': self.uDict, 'ttg': self.bigTTGmat, 'deadline': self.intrDLdist}
        # pickle.dump(lmdpInfo, filehandler)

    def recomputeTTGAndHankelForNewDL(self, goalDLmeans, *newvars):
        if newvars:
            self.goalDLvars = newvars[0]

        self.goalDLmeans = goalDLmeans
        self.goalDLpmfs= []
        self.goalHankels= []
        for i in range(len(goalDLmeans)):
            dist = sp.stats.norm(self.goalDLmeans[i], self.goalDLvars[i])
            dl_pmf = self.getDeadlinePMF(dist, addHeavyTail=True)
            goal_hankel = hankel(dl_pmf)
            self.goalDLpmfs.append(dl_pmf)
            self.goalHankels.append(goal_hankel)





    def getPrReachOverIntervalGivenTime(self, t_from, t_to, pmf):
        # interval is a tuple (t_from, t_to) that we integrate over.  t_to is NOT included
        if t_to > pmf.size:
            t_to = pmf.size - 1
        if t_from > pmf.size:
            t_from = pmf.size - 1

        intervalpmf = np.zeros(pmf.size)
        intervalpmf[np.arange(t_from, t_to)] = pmf[np.arange(t_from, t_to)]
        h = hankel(intervalpmf)
        sucReachBig = np.dot(h[t_from, :], self.bigTTGmat.transpose())
        return sucReachBig

    def getBigState(self, polnumber, state):
        return self.bigIdxList[polnumber] + np.array(state)

    def getPolicy(self, polnumber):
        idxs = np.linspace(0, len(self.bigZ), len(self.bigZ) / self.ss.ns + 1).astype(int)
        i = polnumber
        return self.bigU[np.ix_(range(idxs[i], idxs[i + 1]), range(idxs[i], idxs[i + 1]))]

    def getStateConditionedTransition(self, polnumber, currentstate):
        idxs = np.linspace(0, len(self.bigZ), len(self.bigZ) / self.ss.ns + 1).astype(int)
        i = self.ss.validList.index(polnumber)
        polslice = np.arange(idxs[i], idxs[i + 1])
        state = idxs[i] + currentstate
        return self.bigU[state, :]

    def getHankel(self, dlpmfs):
        return np.array([hankel(dlpmfs[i]) for i in range(dlpmfs.shape[0])])

    def createCostFunction(self, termLISet, wallList, baseCost):
        cfunction = np.ones(self.ss.ns) * np.exp(-baseCost)
        cfunction[termLISet] = 1
        cfunction[wallList] = 0  # Order matters here, we want to overwrite goal states that are in walls.
        return cfunction

    def computeZFunction(self, Q, P):
        # Probably a cleaner way to write this.
        QP = csr.dot(Q, P)
        zold = np.ones(QP.shape[1], dtype=np.double)
        diff = 10000
        iter = 0
        while iter < 100:
            znew = QP.dot(zold)
            znew = znew/znew.max()
            diff = np.sum(np.abs(zold-znew))
            zold = znew
            iter += 1

        znew[znew == 0] = znew[np.where(znew > 0)].min()
        return znew

    def computePolicy(self, z, p):
        z = csr(z)
        normalizer = sp.sparse.diags(np.divide(1, p.dot(z.transpose()).toarray()).squeeze(), dtype=np.double)
        scaled = p.multiply(z)
        pol = sp.dot(normalizer, scaled)
        return pol

    def computeTimeToGo(self, u, terminal):
        # This returns an N^2 sized distribution where each element is a Gaussian RV of the hitting time from x_i to x_j
        # under pi_j.  pi_j indexes the blocks of the vector, each of size N.  It may be more accurate to use
        # Gamma Distributions in the future as they can be made guarantee that the hitting time won't be less than the
        # allowed hitting time under the actual matrix operations.

        # Calculate 1st moment
        arbHighTimeVal = 1E6
        terminal = tuple(terminal)
        Q = delete_from_csr(u, terminal, terminal)
        N = csr(
            sp.sparse.identity(Q.shape[0], dtype=np.double) - Q + sp.sparse.diags(np.random.rand(Q.shape[0]) * 1E-6))
        self.t_mean = spsolve(N, np.ones(Q.shape[0]))
        self.t_mean[np.abs(
            self.t_mean) > 1E4] = arbHighTimeVal  # This sets any states that isolated from the goal state to an arbitrarily high ttg value.
        # Calculate 2nd moment
        b = spsolve(N, self.t_mean)
        self.t_var = 2 * b - self.t_mean - np.power(self.t_mean, 2)
        self.t_var[np.abs(
            self.t_var) > 1E4] = 0.01  # For states that are isolated from the goal state, we set the variance to low, so it is represented as having a very certain high arrival time.
        self.t_var[self.t_var < 1E-4] = 0.001
        # self.t_var = np.ones(self.t_mean.shape)*0.01

        # Add back in terminal state hitting time
        for i in terminal:
            self.t_mean = np.insert(self.t_mean, i, 0)
            self.t_var = np.insert(self.t_var, i, 0.001)

        # t_mean = np.insert(t_mean, terminal, 0)
        # t_var = np.insert(t_var, terminal, 0.001)

        # self.maxTime = int(np.round(self.t_mean[self.t_mean < arbHighTimeVal].max() + 20))

        nDist = norm(self.t_mean, self.t_var)
        timeSpan = np.arange(0, self.maxTime)
        nDist = norm(self.t_mean, self.t_var)

        return nDist

    def getDeadlinePMF(self, intrDLdist, addHeavyTail):
        # addHeavyTail is for adding a small but constant probability mass to the deadline function.  This is used so
        # that there is a small gradient in the value function that is propagated and we won't have to break ties when
        # argmax is called to compute the best policy

        deadlinePMFList = [intrDLdist.cdf(0.5)] + \
                          [intrDLdist.cdf(x) - intrDLdist.cdf(x - 1) for x in
                           np.arange(1.5, self.maxTime - 0.5)] + \
                          [intrDLdist.cdf(10000) - intrDLdist.cdf(self.maxTime - 0.5)]

        deadlinePMF = np.array(deadlinePMFList).transpose()
        if addHeavyTail:
            deadlinePMF = deadlinePMF + np.ones(deadlinePMF.size)*0.005
            deadlinePMF = deadlinePMF/deadlinePMF.sum()
        return deadlinePMF

    def getPReachGiveDeadline(self, dlpmf, ttgtensor):
        pdfcdfmult = np.multiply(np.expand_dims(dlpmf, 1), ttgtensor)
        psa = pdfcdfmult.sum(2)
        return psa

# Setup is used for drawing the environment.  Walls, goal points.
class Setup(arcade.Window):
    def __init__(self, width, height, goalstates, buttonstates, initGrid, buttonToWall):
        """
        Set up the application.
        """
        self.wflag = False
        self.gflag = False
        self.bflag = False  # Button Flag
        self.hflag = False
        self.bSelectFlag = False
        self.selectedButton = None
        super().__init__(width, height, title="Gridworld")

        self.ss = StateSpace(ROW_COUNT, COLUMN_COUNT, wallMatrix=np.zeros(0), buildTM=False)
        self.wallMatrix = np.zeros([3, 3])
        self.goalStates = goalstates
        # self.player = Agent(COLUMN_COUNT//2, ROW_COUNT//2, arcade.color.RED, wmat=np.zeros([ROW_COUNT, COLUMN_COUNT]))
        self.player = Agent(2, 2, arcade.color.RED, wmat=np.zeros([ROW_COUNT, COLUMN_COUNT]))
        self.left_down = False
        arcade.set_background_color(arcade.color.LIGHT_SLATE_GRAY)

        X = np.arange(0, COLUMN_COUNT, 1)
        Y = np.arange(0, ROW_COUNT, 1)
        X, Y = np.meshgrid(X, Y)

        self.draw_time = 0
        self.shape_list = None

        # Create a 2 dimensional array.
        self.grid = initGrid
        self.buttonStates = buttonstates
        self.buttonToWall = buttonToWall

        arcade.set_background_color(arcade.color.BLACK)

    def setup(self):
        self.shape_list = arcade.ShapeElementList()
        # self.shape_list_walls = arcade.ShapeElementList()

        # --- Create all the rectangles

        # We need a list of all the points and colors
        self.point_list = []
        self.color_list = []
        self.color_list_walls = []

        # Now calculate all the points
        for x in range(0, SCREEN_WIDTH, SQUARE_SPACING):
            for y in range(0, SCREEN_HEIGHT, SQUARE_SPACING):
                # Calculate where the four points of the rectangle will be if
                # x and y are the center
                top_left = (x, y + SQUARE_HEIGHT)
                top_right = (x + SQUARE_WIDTH, y + SQUARE_HEIGHT)
                bottom_right = (x + SQUARE_WIDTH, y)
                bottom_left = (x, y)

                # Add the points to the points list.
                # ORDER MATTERS!
                # Rotate around the rectangle, don't append points caty-corner
                fourpnts = [top_left, top_right, bottom_right, bottom_left]
                self.point_list.append(fourpnts)
                # self.point_list.append(top_left)
                # self.point_list.append(top_right)
                # self.point_list.append(bottom_right)
                # self.point_list.append(bottom_left)

                # Add a color for each point. Can be different colors if you want
                # gradients.
                for i in range(4):
                    self.color_list.append(arcade.color.LIGHT_SLATE_GRAY)
                    self.color_list_walls.append(arcade.color.BLACK)

        self.point_list_flat = [pnt for wset in self.point_list for pnt in wset]
        self.shape_background = arcade.create_rectangles_filled_with_colors(self.point_list_flat, self.color_list)
        # shape_walls = arcade.create_rectangles_filled_with_colors(self.point_list, self.color_list_walls)
        # self.shape_list.append(shape)
        # self.shape_list_walls.append(shape_walls)

    def add_to_shape_list(self, row, col):
        # We need a list of all the points and colors
        point_list = []
        color_list = []

        top_left = (col, row + SQUARE_HEIGHT)
        top_right = (col + SQUARE_WIDTH, row + SQUARE_HEIGHT)
        bottom_right = (col + SQUARE_WIDTH, row)
        bottom_left = (col, row)

        point_list.append(top_left)
        point_list.append(top_right)
        point_list.append(bottom_right)
        point_list.append(bottom_left)

        for i in range(4):
            color_list.append(arcade.color.BLACK)

        shape = arcade.create_rectangles_filled_with_colors(point_list, color_list)
        self.shape_list.append(shape)

    def on_draw(self):
        """
        Render the screen.
        """
        self.shape_list = arcade.ShapeElementList()
        wallCoordsFlat = self.grid.transpose().flatten().nonzero()[0]
        # This command has to happen before we start drawing

        wallblack = [arcade.color.BLACK]*4
        backgroundcolor = [arcade.color.LIGHT_GRAY]*4

        modded_color_list = [wallblack if i in wallCoordsFlat else backgroundcolor for i in range(len(self.point_list))]
        modded_color_list = [i for j in modded_color_list for i in j]
        modded_shape_with_walls = arcade.create_rectangles_filled_with_colors(self.point_list_flat, modded_color_list)
        self.shape_list.append(modded_shape_with_walls)

        arcade.start_render()

        # --- Draw all the rectangles
        self.shape_list.draw()

        x = (SQUARE_SPACING * self.player.x + SQUARE_SPACING // 2)
        y = (SQUARE_SPACING * self.player.y + SQUARE_SPACING // 2)
        # arcade.draw_rectangle_filled(x, y, HALF_SQUARE_WIDTH, HALF_SQUARE_HEIGHT, self.player.color)

        wallCoords = self.grid.nonzero()

        startsquarex = (SQUARE_SPACING * STARTX + SQUARE_SPACING // 2)
        startsquarey = (SQUARE_SPACING * STARTY + SQUARE_SPACING // 2)
        arcade.draw_rectangle_filled(startsquarex, startsquarey, SQUARE_WIDTH, SQUARE_HEIGHT,
                                     arcade.color.LIGHT_SLATE_GRAY)

        # for i in range(len(wallCoords[0])):
        #     x = (SQUARE_SPACING * wallCoords[1][i]) + HALF_SQUARE_WIDTH
        #     y = (SQUARE_SPACING * wallCoords[0][i]) + HALF_SQUARE_HEIGHT
        #     arcade.draw_rectangle_filled(x, y, SQUARE_WIDTH, SQUARE_HEIGHT, arcade.color.BLACK)
        buttonCoords = self.buttonStates.flatten().nonzero()[0]
        for i in buttonCoords:
            row, col = self.ss.liToCoord[i]
            if self.buttonStates.flat[i] == 1:
                x = (SQUARE_SPACING * col) + HALF_SQUARE_WIDTH
                y = (SQUARE_SPACING * row) + HALF_SQUARE_HEIGHT
                arcade.draw_rectangle_filled(x, y, SQUARE_WIDTH, SQUARE_HEIGHT, arcade.color.YELLOW)
            if self.buttonStates.flat[i] == 2:
                x = (SQUARE_SPACING * col) + HALF_SQUARE_WIDTH
                y = (SQUARE_SPACING * row) + HALF_SQUARE_HEIGHT
                arcade.draw_rectangle_filled(x, y, SQUARE_WIDTH, SQUARE_HEIGHT, arcade.color.LIGHT_YELLOW)

        for i in self.goalStates:
            row, col = self.ss.liToCoord[i]
            x = (SQUARE_SPACING * col) + HALF_SQUARE_WIDTH
            y = (SQUARE_SPACING * row) + HALF_SQUARE_HEIGHT
            arcade.draw_rectangle_filled(x, y, SQUARE_WIDTH, SQUARE_HEIGHT, arcade.color.INDIA_GREEN)

        for i in self.buttonToWall[self.selectedButton]:
            row, col = self.ss.liToCoord[i]
            x = (SQUARE_SPACING * col) + HALF_SQUARE_WIDTH
            y = (SQUARE_SPACING * row) + HALF_SQUARE_HEIGHT
            arcade.draw_rectangle_filled(x, y, SQUARE_WIDTH, SQUARE_HEIGHT, arcade.color.GRAY)

    def on_mouse_press(self, x, y, button, modifiers):
        """
        Called when the user presses a mouse button.
        """
        # Change the x/y screen coordinates to grid coordinates
        column = x // SQUARE_SPACING
        row = y // SQUARE_SPACING
        li = self.ss.coordDict[row, column]
        print(f"Click coordinates: ({x}, {y}). Grid coordinates: ({row}, {column})")
        iswall = self.grid[row, column]
        isbutton = self.buttonStates[row, column]
        # Make sure we are on-grid. It is possible to click in the upper right
        # corner in the margin and go to a grid location that doesn't exist
        if row < ROW_COUNT and column < COLUMN_COUNT:

            # Flip the location between 1 and 0.
            if self.wflag:
                if self.grid[row, column] == 0:
                    self.grid[row, column] = 1
                else:
                    self.grid[row, column] = 0
                # self.wFLAG = False

            if self.gflag:
                self.goalStates.remove(li) if li in self.goalStates else self.goalStates.add(li)
                # self.gFLAG = False

            if self.bflag:
                if self.buttonStates[row, column] == 0:
                    self.buttonStates[row, column] = 1
                elif self.buttonStates[row, column] == 1:
                    self.buttonStates[self.buttonStates == 2] = 1
                    self.buttonStates[row, column] = 2
                    self.selectedButton = li
                    self.bSelectFlag = True
                elif self.buttonStates[row, column] == 2:
                    self.buttonStates[row, column] = 0
                    self.selectedButton = None
                    self.bSelectFlag = False

                # self.bFLAG = False

            if self.bSelectFlag and not isbutton:
                if li not in self.buttonToWall[self.selectedButton]:
                    self.buttonToWall[self.selectedButton].append(li)
                else:
                    idx = self.buttonToWall[self.selectedButton].index(li)
                    del self.buttonToWall[self.selectedButton][idx]



    def update(self, dt):
        """ Move everything """
        self.player.moveForInput()


    def on_key_press(self, key, modifiers):
        """
        Called whenever a key is pressed.
        """
        if key == arcade.key.Q:
            wallMatrix = self.grid
            self.wallMatrix = self.grid
            arcade.window_commands.close_window()

        if key == arcade.key.W:
            self.wflag = True

        if key == arcade.key.G:
            self.gflag = True

        if key == arcade.key.B:
            self.bflag = True

        if key == arcade.key.H:
            self.hflag = True

    def on_key_release(self, key, modifiers):
        """
        Called when the user releases a key.
        """
        if key == arcade.key.UP or key == arcade.key.DOWN:
            self.player.delta_y = 0
        elif key == arcade.key.LEFT or key == arcade.key.RIGHT:
            self.player.delta_x = 0

        if key == arcade.key.W:
            self.wflag = False

        if key == arcade.key.G:
            self.gflag = False

        if key == arcade.key.B:
            self.bflag = False

        if key == arcade.key.H:
            self.hflag = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.wflag:
            column = x // SQUARE_SPACING
            row = y // SQUARE_SPACING

            print(f"Click coordinates: ({x}, {y}). Grid coordinates: ({row}, {column})")

            # Make sure we are on-grid. It is possible to click in the upper right
            # corner in the margin and go to a grid location that doesn't exist
            if row < ROW_COUNT and column < COLUMN_COUNT:

                # Flip the location between 1 and 0.
                if self.grid[row, column] == 0:
                    self.grid[row, column] = 1


class Environment:

    def __init__(self, ss, wallVectorList, nP, nV, nW, lmdp):
        self.ss = ss
        self.wallVecList = wallVectorList
        self.nP = nP
        self.nV = nV
        self.nW = nW
        self.lmdp = lmdp

    def getRayCastDataForState(self, vstate, parset):
        aset = parset[1, :]
        rs_set = aset.reshape(self.nV * self.nP, self.nW)
        hits = np.logical_and(np.all(parset >= 0, axis=0), np.all(parset <= 1, axis=0))
        rs_hits = hits.reshape(self.nP * self.nV, self.nW)
        hitInds = np.where(np.any(rs_hits == True, axis=1) == True)[0]
        masked = rs_set[hitInds, :] * rs_hits[hitInds, :]
        masked[masked == 0] = 1
        minAlphas = np.min(masked, axis=1)

        oneVec = np.ones(self.nP * self.nV)
        oneVec[hitInds] = minAlphas

        rayAs = np.repeat(np.array(self.ss.validCoords), self.nP, axis=0).astype('double') + 0.5
        rayBs = np.tile(self.ss.perimCoords, (self.nV, 1)) + 0.5
        BA = rayBs-rayAs

        finalRays = rayAs + np.multiply(BA, oneVec[:, None])

        template = np.tile(np.expand_dims(np.linspace(0, 1, 1001), axis=0), (finalRays.shape[0], 1))
        ydiff = np.abs(np.abs(rayAs[:, 0] - finalRays[:, 0]))
        xdiff = np.abs(np.abs(rayAs[:, 1] - finalRays[:, 1]))

        tempy = np.multiply(template, ydiff[:, None])
        tempx = np.multiply(template, xdiff[:, None])

        # ytemplate = np.tile(np.expand_dims(np.linspace(0, 1, 1001), axis=0), finalRays.shape[0])

        repFR = np.repeat(finalRays, COLUMN_COUNT*ROW_COUNT)
        ys = np.arange(0, ROW_COUNT)
        topy = np.repeat(ROW_COUNT)
        xs = np.arange(0, COLUMN_COUNT)
        yx = ys-xs

        yst = np.tile(ys, ())

        return []



class MyGame(arcade.Window):
    """
    Main application class.
    """

    def __init__(self, width, height, wallmatset, timeStepArray, goalstates, goalDLmeans, goalDLvars, intrinsicDLmean,
                 intrinsicDLvar, periodToEnvMap, buttonstatelist, allgoallist, buttonPowSet, map2worldIdx,
                 goalmasterdict, loadenvfilename, loadEnvData, saveEnvData, saveenvfilename):
        """
        Set up the application.
        """

        super().__init__(width, height, title="Gridworld")
        super().set_update_rate(0.17)
        self.spaceFlag = False
        self.initFlag = False
        # self.timeStepArray = np.array([4, 8, 4])
        self.currentTime = 0
        self.currentPeriod = 0

        self.intrinsicDLmean = intrinsicDLmean
        self.intrinsicDLvar = intrinsicDLvar

        self.goalStates = np.sort(list(goalstates))
        self.goalDLmeans = goalDLmeans
        # self.goalDLmeans = [30, 17, 69, 59, 90, 90]
        self.goalDLvars = goalDLvars


        self.allgoallist = allgoallist
        self.goalMasterDict = goalmasterdict
        self.buttonPowSet = buttonPowSet
        self.map2worldIdx = map2worldIdx

        self.mvmntDLrefMean = np.max(self.goalDLmeans)  # We set mvmntDLrefMean to be high so that the most extreme case, one period is the length of the entire horizon

        self.periodToEnvMap = periodToEnvMap
        self.buttonStateSet = buttonstatelist
        self.envList = []
        self.universeEnvs = {}
        arcade.set_background_color(arcade.color.DARK_SLATE_GRAY)
        self.wallMatrixSet = wallmatset

        # Compute or load LMDP solutions
        if loadEnvData:
            self.universeEnvs = try_to_load_as_pickled_object_or_None(loadenvfilename)

            # for worldidx in self.universeEnvs:
            #     for i in range(len(self.universeEnvs[worldidx])):
            #         self.universeEnvs[worldidx][i].lmdp.recomputeTTGAndHankelForNewDL([30, 17, 69, 59, 90, 90], [1.2]*len(self.allgoallist))
            #         # self.envList[i].lmdp.recomputeTTGAndHankelForNewDL([30, 16, 59, 49], [1.2, 1.2, 1.2, 1.2])

        else:
            self.computeLMDPData()
            if not os.path.isfile(saveenvfilename):
                save_as_pickled_object(self.universeEnvs, saveenvfilename) if saveEnvData else None

        self.timeStepArray = timeStepArray
        lst = [[i] * self.timeStepArray[i] for i in np.arange(len(self.timeStepArray))] + [[len(self.timeStepArray)-1]] * 5000
        self.periodlist = [i for j in lst for i in j]
        arr = np.array([0] + list(self.timeStepArray.copy()))

        # self.periodInitTimes = np.cumsum(np.concatenate((arr, np.array([1000000]))))
        self.periodInitTimes = np.cumsum(arr)

        self.ns = COLUMN_COUNT*ROW_COUNT
        self.ng = len(self.allgoallist)
        self.np = len(np.unique(self.periodlist))
        self.cWorld = 0
        self.cEnv = self.universeEnvs[self.cWorld][0]  # Current Environment

        # self.agent = Agent(COLUMN_COUNT//2, ROW_COUNT//2, arcade.color.RED, self.cEnv.ss.wallMatrix, self.cEnv.ss)
        self.agent = Agent(2, 2, arcade.color.RED, wmat=np.zeros([ROW_COUNT, COLUMN_COUNT]))
        self.universeReachMaps = {}

        for worldIdx in self.universeEnvs:
            self.computeReachabilityMaps(worldIdx)

        # string1 = [2, 0, 1, 3]  # Simple example in paper L->R
        # string2 = [0, 2, 1, 3]  # Simple example in paper L->R
        # string3 = [2, 0, 3, 1]  # Simple example in paper L->R
        # string4 = [0, 2, 3, 1]  # Simple example in paper L->R

        # string1 = [1, 3, 2, 0]  # Simple example in paper R->L
        # string2 = [1, 3, 0, 2]  # Simple example in paper R->L
        # string3 = [3, 1, 2, 0]  # Simple example in paper R->L
        # string4 = [3, 1, 0, 2]  # Simple example in paper R->L

        # string1 = [0, 1, 3, 2]  # Simple example in paper B->T
        # string2 = [0, 1, 2, 3]  # Simple example in paper B->T
        # string3 = [1, 0, 3, 2]  # Simple example in paper B->T
        # string4 = [1, 0, 2, 3]  # Simple example in paper B->T

        # string1 = [5, 1, 0, 4, 2, 3]
        # string2 = [5, 4, 3, 2, 1, 0]
        # string3 = [5, 4, 3, 2, 1, 0]
        # string4 = [4, 2, 5, 3, 1, 0]
        # string2 = [3, 2, 0, 1]
        # string3 = [2, 3, 1, 0]
        # string4 = [2, 3, 0, 1]
        # string4 = [2, 3, 0, 1]
        # string4 = [2, 3, 0, 1]
        # string4 = [2, 3, 0, 1]
        # string4 = [2, 3, 0, 1]

        stringList = [string1, string2, string3, string4]

        plan = self.computeBestPlan(stringList)

        # plan = self.traceList[np.argmax(self.traceProbs)]
        stichSetForPlan = self.stitchSetList[np.argmax(self.traceProbs)]
        self.agent.initializePlan(self.allgoallist, plan, stichSetForPlan, plan[0], stichSetForPlan[0])

        #
        # self.agent.plan = self.traceList[np.argmax(self.traceProbs)]
        # self.agent.stichSetForPlan = self.stitchSetList[np.argmax(self.traceProbs)]
        self.agent.worldIdx = 0
        # self.agent.piSetList = pi_list
        self.agent.currentGoal = self.agent.plan[0]
        self.agent.currentPolicy = self.agent.stichSetForPlan[0]
        self.cEnv = self.universeEnvs[0][0]


        # Universe is a dictionary for composite world constraint maps.  Each member of universe has a set of
        # reachability maps for that particular world


    def computeReachabilityMaps(self, worldIdx):
        vseeds = np.array([onehot(i, self.ns) for i in self.allgoallist])
        # REARRANGES INDICES SO THAT ROW OF MATRIX IS x_i --> x_j via pi_j
        # rearrageidxs = np.array([self.envList[0].lmdp.getBigState(np.arange(self.ns), i) for i in range(self.ns)]).flatten()
        mvmntTimeIdxs = np.array(list(np.subtract(self.mvmntDLrefMean, self.timeStepArray)) + list([0])) # These times are relative to the mvmntDLrefMean
        world = self.universeEnvs[worldIdx]
        feasibility_list = []  # This stores each v_list for each period [period][goal]
        pi_period_list = []  # [period][goal]
        v_mat = [np.zeros(self.ns)]*self.ng  # Stores a v vector for all goals

        for i in np.flip(np.arange(len(self.periodInitTimes)-1)):
            t_from = self.periodInitTimes[i]
            t_to = min(self.periodInitTimes[i+1], 200)
            print('t_from: ' + str(self.periodInitTimes[i]))
            print('t_to: ' + str(self.periodInitTimes[i+1]))
            self.cEnv = world[self.periodToEnvMap[i]]
            goalDLpmfs = self.cEnv.lmdp.goalDLpmfs
            mvmntpmf = [self.cEnv.lmdp.movementDLPMF]

            m_source, v_source_new_set = self.getBackPropInfoAllGoals(vseeds, t_from, t_to, goalDLpmfs, 'source')
            m_prop, v_prop_new_set = self.getBackPropInfoAllGoals(v_mat, mvmntTimeIdxs[i], self.mvmntDLrefMean + 1, mvmntpmf, 'prop')

            v_new_mat, pi_list = self.getAllPolicyAndValueSets(m_source, m_prop, v_mat)
            feasibility_list.append(v_new_mat)
            pi_period_list.append(pi_list)
            v_mat = v_new_mat
            # for v in v_new_mat:
            #     plot(v, 1)

            print(i)

        feasibility_list.reverse()
        pi_period_list.reverse()
        feasibility_list.append([np.zeros(self.ns)] * self.ng)
        pi_period_list.append([np.zeros(self.ns)] * self.ng)
        # vTensor = np.ndarray(feasibility_list)

        # plot(feasibility_list[0][0])

        # Each matrix in pol_switch_period_list[period][goal] will give the matrix that has off diagonals that send the
        # agent to the proper policy in self.bigU
        bigSet = [[[pollist_goal*self.ns]+np.arange(self.ns) for pollist_goal in period] for period in pi_period_list]
        allones = np.ones(self.ns**2)
        rowinds = np.arange(self.ns**2)
        pol_switch_period_list = [[csr((allones, (rowinds, np.tile(np.array(goal_pol).squeeze(), [self.ns]))),
                                       shape=(self.ns ** 2, self.ns ** 2)) for goal_pol in period] for period in bigSet]
        # self.piSet = [polset[1] for polset in pi_list]
        # self.vset.reverse()
        # self.agent.piSet = self.piSet.copy()

        self.agent.timePeriodList = self.timeStepArray
        self.agent.currentSS = self.cEnv.ss

        worldParams = {
            'feasibility': feasibility_list,
            'pi_list': pi_period_list,
            'pol_switcher': pol_switch_period_list}

        self.universeReachMaps[worldIdx] = worldParams




    def computeBestPlan(self, stringList):
        print('Current Time: ' + str(self.currentTime))
        print('Current Period: ' + str(self.currentPeriod))
        print('Current Policy: ' + str(self.agent.currentPolicy))

        # initworld = self.universeEnvs['main']
        initworld = 0

        self.stringList = stringList
        valAndPols = [self.computeTraceSatisfactionValue(trace, initworld) for trace in self.stringList]
        vSetMatrix = np.array([np.array(pair[0]) for pair in valAndPols])

        self.traceProbs = [np.prod(e) for e in vSetMatrix]

        self.stitchSetList = [pair[1] for pair in valAndPols]

        vSetMatrix = np.array([np.array(pair[1]) for pair in valAndPols])
        # self.traceProbs[1]=0

        plan = self.stringList[np.argmax(self.traceProbs)]
        return plan

    def initializeAgentParametersForPlan(self):
        self.agent.plan = self.stringList[np.argmax(self.traceProbs)]
        self.agent.stichSetForPlan = self.stitchSetList[np.argmax(self.traceProbs)]
        self.agent.piSetList = pi_list
        self.agent.currentGoal = self.agent.plan[0]
        self.agent.currentPolicy = self.agent.stichSetForPlan[0]

    def getBackPropInfo(self):
        return []

    def getBestTrace(self, traceList, traceValues):
        return traceList[np.argmax(traceValues)]


    def computeTraceSatisfactionValue(self, trace, initworldidx, tol=0.00):
        # String is a list of goal indexes (not goal state indexes).

        v_at_period_change_list = []
        pi_at_period_change_list = []
        xCur = self.agent.getLIss()
        cur_period = 0
        cur_time = 0
        worldIdx = initworldidx
        for i, gIdx in enumerate(trace):
            timeToNextPeriod, next_period = self.getRemainingTimeUntilNextPeriod(cur_time)

            v_at_change, pi_stitch = self.computeStitch(next_period, timeToNextPeriod, gIdx, xCur, cur_time, worldIdx)  # This is a problem because the next period might be the same period if the goal happens in between the next period and the goal
            pi_list = self.universeReachMaps[worldIdx]['pi_list']
            if v_at_change > tol or cur_time > ROW_COUNT*COLUMN_COUNT-1:
                timeToGoal = self.getTimeFromSingleGoal(xCur, cur_period, cur_time, gIdx, pi_stitch, pi_list, worldIdx)
                xCur = self.allgoallist[gIdx]
                cur_time = np.min([cur_time+timeToGoal, ROW_COUNT*COLUMN_COUNT-1])
                cur_period = self.periodlist[cur_time]
                if self.goalMasterDict[self.allgoallist[gIdx]]['type'] == 'button':
                    # Here we need to preform the buttons operation on the current wallmat and look up the world index
                    buttonwallinds = self.goalMasterDict[self.allgoallist[gIdx]]['walls']
                    newwallmat = wallTransform(buttonwallinds, self.cEnv.ss.wallMatrix)
                    worldIdx = self.map2worldIdx[matstring(newwallmat)]
                    # worldIdx = tuple(np.sort(list(worldIdx)+list(self.allgoallist[gIdx])))
                v_at_period_change_list.append(v_at_change)
                pi_at_period_change_list.append(pi_stitch)
            else:
                v_at_period_change_list.append(v_at_change)
                pi_at_period_change_list.append(pi_stitch)
                break


        v_at_period_change_list = [v_at_period_change_list[i] if i < len(v_at_period_change_list) else 0 for i in range(len(trace))]
        return v_at_period_change_list, pi_at_period_change_list

    def getNextPeriod(self, gIdx, currenttime):
        # Returns the next period or the current on depending if the goal occurs in between the current time and the next period.
        next_period = np.argmax(self.periodInitTimes > currenttime)
        period_of_goal = np.argmax(self.periodInitTimes > self.goalDLmeans[gIdx])
        return []

    def getAgentTimeAtGoals(self, t_init, timeToGoal):
        return t_init + np.array(timeToGoal)

    def getRemainingTimeUntilNextPeriod(self, time):
        next_period = np.argmax(self.periodInitTimes > time)
        nextPeriodTime = self.periodInitTimes[next_period]
        timeUntilNextPeriod = nextPeriodTime - time
        return timeUntilNextPeriod, next_period

    # Problem: the stitch target cannot be the next goal, it has to be the value for the next policy?
    def computeStitch(self, nextPeriod, timeToNextPeriod, nextGoalIdx, currentState, currentTime, worldIdx):
        currentPeriod = nextPeriod - 1
        vseed = onehot(self.allgoallist[nextGoalIdx], self.ns)
        feasibility_list = self.universeReachMaps[worldIdx]['feasibility']
        worldEnvs = self.universeEnvs[worldIdx]
        nextv_for_prop = feasibility_list[nextPeriod][nextGoalIdx]
        cEnv = self.universeEnvs[worldIdx][currentPeriod]
        t_from_source = currentTime
        t_to_source = currentTime + timeToNextPeriod
        t_from_prop = self.mvmntDLrefMean - timeToNextPeriod
        t_to_prop = self.mvmntDLrefMean + 1

        sourcepmf = worldEnvs[self.periodToEnvMap[nextPeriod]].lmdp.goalDLpmfs[nextGoalIdx]
        mvmntpmf = worldEnvs[self.periodToEnvMap[nextPeriod]].lmdp.movementDLPMF

        pr_reach_before_dl_source = cEnv.lmdp.getPrReachOverIntervalGivenTime(t_from_source, t_to_source, sourcepmf)
        pr_reach_before_dl_prop = cEnv.lmdp.getPrReachOverIntervalGivenTime(t_from_prop, t_to_prop, mvmntpmf)
        rs_prbd_source = pr_reach_before_dl_source.reshape(self.ns, self.ns)
        rs_prbd_prop = pr_reach_before_dl_prop.reshape(self.ns, self.ns)

        # Propagate

        multmat_source = np.multiply(rs_prbd_source, vseed[:, np.newaxis])
        multmat_prop = np.multiply(rs_prbd_prop, nextv_for_prop[:, np.newaxis])

        matmul_combined = multmat_source + multmat_prop

        v_stitch_source = np.max(multmat_source, axis=0)
        v_stitch_prop = np.max(multmat_prop, axis=0)
        v_stitch = np.max(matmul_combined, axis=0)  # TODO: Figure out if "flat" looses information.
        pi_stitch = np.argmax(matmul_combined, axis=0)

        v_stitch_at_cur_state = v_stitch[currentState]
        pi_stitch_at_cur_state = pi_stitch[currentState]
        u = self.cEnv.lmdp.getPolicy(17)
        # polquiver(u, self.cEnv.ss.liToCoord, self.cEnv.ss.wallList, self.cEnv.ss.wallMatrix, valmap=v_stitch)
        return v_stitch_at_cur_state, pi_stitch_at_cur_state


    def getTimeFromSingleGoal(self, x0, init_period, init_time, goalIdx, pi_stitch, pi_list, worldIdx):
        cur_period = init_period
        worldEnvs = self.universeEnvs[worldIdx]
        curEnv = worldEnvs[cur_period]
        curPol = pi_stitch if pi_stitch else pi_list[cur_period][goalIdx][x0]
        x = csr(onehot(curPol * self.ns + x0, self.ns ** 2))
        curU = curEnv.lmdp.bigU
        maxT = COLUMN_COUNT*ROW_COUNT*2
        timeElapsed = 0
        for t in range(init_time, maxT):
            smallstatevec = np.asarray(x.todense())[0].reshape(self.ns, self.ns).sum(axis=0)
            collapsedPolVec = np.asarray(x.todense())[0].reshape(self.ns, self.ns).sum(axis=1)
            if smallstatevec[self.allgoallist[goalIdx]] > 0.98:
                break

            if cur_period != self.periodlist[t]:
                cur_period = self.periodlist[t]
                curEnv = worldEnvs[cur_period]
                curU = curEnv.lmdp.bigU
                polswitcher = self.universeReachMaps[worldIdx]['pol_switcher']
                x = x.dot(polswitcher[cur_period][goalIdx])  # Map the agent to the correct policy
            x_new = x.dot(curU)
            # print(cur_period)
            # test = np.asarray(x_new.todense()).squeeze()
            # rs = test.reshape(self.ns, self.ns)
            # summed = rs.sum(axis=0)
            # plot(summed)

            x = x_new
            timeElapsed += 1

        return timeElapsed

    def getTieBreaker(self, goalInd, t_mean):
        indStart = self.ns * goalInd
        indEnd = indStart + self.ns
        tiebreaker = (1 / (t_mean[indStart:indEnd] + 1)) * 1E-7
        return tiebreaker

    def getBackPropInfoAllGoals(self, vmat, t_from, t_to, pmfs, bp_type):
        mset, vnset, piset = [], [], []
        for i, v in enumerate(vmat):
            pmf = pmfs[0] if bp_type == 'prop' else pmfs[i]
            mDict, v_new = self.getBackwardPropegationInfo(t_from, t_to, pmf, v, i, bp_type)
            mset.append(mDict)
            vnset.append(v_new)
        return mset, vnset

    def getBackwardPropegationInfo(self, t_from, t_to, pmf, v_in, goalIdx, bp_type):
        if bp_type is 'source':
            pr_reach_before_dl = self.cEnv.lmdp.getPrReachOverIntervalGivenTime(t_from, t_to, pmf)
            rs_prbd = pr_reach_before_dl.reshape(self.ns, self.ns)
            tiebreakmat = (1/(self.cEnv.lmdp.t_mean + 1) * 1E-7).reshape(self.ns, self.ns)
            multmat_source = np.multiply(rs_prbd, v_in[:, np.newaxis])+tiebreakmat
            # tiebreaker = self.getTieBreaker(self.goalStates[goalIdx], self.cEnv.lmdp.t_mean)

            v_out_new = np.max(multmat_source + tiebreakmat, axis=0)
            mDict = {'fullSource': multmat_source}

            m_out = multmat_source

        elif bp_type is 'prop':
            # v_flat = v_in.copy()
            # v_flat[v_flat > 1] = 1
            pr_reach_before_dl = self.cEnv.lmdp.getPrReachOverIntervalGivenTime(t_from, t_to, pmf)
            rs_prbd = pr_reach_before_dl.reshape(self.ns, self.ns)
            tiebreakmat = (1 / (self.cEnv.lmdp.t_mean + 1) * 1E-7).reshape(self.ns, self.ns)
            multmat = np.multiply(rs_prbd, v_in[:, np.newaxis]) + tiebreakmat
            # multmat_full = np.multiply(rs_prbd, v_in[:, np.newaxis])
            v_out_new = np.max(multmat, axis=0)  # TODO: Figure out if "flat" looses information.
            # mDict = {'fullProp': multmat, 'flatProp': multmat_flat}
            # mDict = {'fullProp': multmat}
            m_out = multmat

        return m_out, v_out_new

    def getAllPolicyAndValueSets(self, mset_source, mset_prop, v_list):
        v_new_list = []
        pi_list = []
        v_flat = np.array(v_list.copy())
        v_flat[v_flat > 1] = 1
        for i in range(len(mset_prop)):
            multed_combined = mset_source[i] + mset_prop[i]
            # multed_combined_flat = mset_source[i]['fullSource'] + mset_prop[i]
            # v_new_list.append(np.max(multed_combined_flat, axis=0) + v_list[i])  # This version is probably bad
            v_new_list.append(np.max(multed_combined, axis=0))
            pi_list.append(np.argmax(multed_combined, axis=0))

        return v_new_list, pi_list

    def computeLMDPData(self):
        for worldkey in self.wallMatrixSet:
            wallMatrixSet = self.wallMatrixSet[worldkey]
            envList = []
            for wallmat in wallMatrixSet:
                ss = StateSpace(ROW_COUNT, COLUMN_COUNT, buildTM=True, wallMatrix=wallmat)
                walls = self.getWallVectorList(wallmat)
                nP = len(ss.perimCoords)
                nV = len(ss.validCoords)
                nW = len(walls)
                baseCost = 10
                # rays = self.computeRays(self, ss, walls)

                lmdpkwargs = {
                    'stateSpace': ss,
                    'baseCost': baseCost,
                    'intrinsicDLmean': self.intrinsicDLmean,
                    'intrinsicDLvar': self.intrinsicDLvar,
                    'goalDLmeans': self.goalDLmeans,
                    'goalDLvars': self.goalDLvars,
                    'mvmntDLrefMean': self.mvmntDLrefMean
                }

                # lmdp = LMDP(ss, termCoords='Valid', baseCost=10, intrinsicDLmean=self.intrinsicDeadline, intrinsicDLvar=self.intrinsicDLvar)
                lmdp = LMDP(**lmdpkwargs)

                envList.append(Environment(ss, walls, nP, nV, nW, lmdp))

            self.universeEnvs[worldkey] = envList

    # TODO: Make sure that the vstate is converted into the correct LI for params
    def computeRays(self, ss, walls):
        #  Compute ray-wall intersections with a(B-A)-b(D-C)=(C-A) => [BA|DC][a,-b]^T = CA
        nP = len(ss.perimCoords)
        nV = len(ss.validCoords)
        nW = len(walls)

        # TODO write code for when nW == 0
        if nW > 0:
            bigA = np.repeat(np.array(ss.validCoords), nW * nP, axis=0).flatten().astype('double') + 0.5
            bigB = np.tile(np.repeat(np.array(ss.perimCoords), nW, axis=0), (nV, 1)).flatten() + 0.5
            c = np.array([w[0] for w in walls])
            d = np.array([w[1] for w in walls])
            bigC = np.tile(c, (nV * nP, 1)).flatten() + 1e-6  # Prevent matrix degeneracy with +epsilon
            bigD = np.tile(d, (nV * nP, 1)).flatten()

            bigV = (bigB - bigA).astype('double')

            # Add small delta = 0.01 to prevent matrix degeneracy
            rv = bigV.reshape(bigV.shape[0] // 2, 2)
            rv[np.all(rv == 0, axis=1)] = [0.01, 0.01]
            bigV = rv.flatten()

            bigW = (bigD - bigC).astype('double')
            # rw = bigW.reshape(bigW.shape[0] // 2, 2)
            bvec = bigC - bigA

            rowindsForV = np.arange(0, bigV.shape[0]).tolist()
            colindsForV = [i for i in np.arange(0, bigV.shape[0], 2).repeat(2)]
            rowindsForW = np.arange(0, bigW.shape[0]).tolist()
            colindsForW = [i for i in np.arange(1, bigW.shape[0], 2).repeat(2)]
            row = rowindsForV + rowindsForW
            col = colindsForV + colindsForW

            spMat = sp.sparse.coo_matrix((np.array([bigV, bigW]).flatten(), (row, col)))
            csrMat = csr(spMat)

            params = sp.sparse.linalg.spsolve(csrMat, bvec)

            aParams = params[np.arange(0, len(params), 2)]
            bParams = params[np.arange(1, len(params), 2)] * -1
            paramSet = np.array([aParams, bParams])

            # nBlockInds = np.arange(0, nV * nP * nW, nP * nW)
            # kBlockInds = np.arange(0, nP * nW, nP)
            # mBlockInds = np.arange(0, nW)

        return paramSet

    def getWallVectorList(self, wallMatrix):
        wallLineList = []
        wallLineList = [[[[y, x], [y, x + 1]], [[y, x + 1], [y + 1, x + 1]], [[y + 1, x + 1], [y + 1, x]], [[y + 1, x], [y, x]]] for x
                        in range(COLUMN_COUNT) for y in range(ROW_COUNT) if wallMatrix[y, x] == 1]
        flatWLL = [x for y in wallLineList for x in y]
        flat2 = [x[0] + x[1] for x in flatWLL]
        unique_lines = [list(x) for x in set(tuple(x) for x in flat2)]
        walls = [[[x[0], x[1]], [x[2], x[3]]] for x in unique_lines]

        return walls

    # Notation. Ray is defined by two vectors start vector a, end vector b.  walls is a collection of start/end vectors
    # for each line segment that defines a wall with the notation convention start: c, end: d.
    def getCollisionMat(self, ray, walls):
        ba = ray[1]-ray[0]  #b-a
        blks = [np.array([ba, np.array(w[1])-np.array(w[0])]).transpose() for w in walls]  # Small blkmat [b-a,d-c]
        c = np.array([w[0] for w in walls]).flatten()
        ca = c-np.tile(ray[0], len(walls))  #c-a
        bdiag = sp.linalg.block_diag(*blks)

        return bdiag, ca

    def setup(self):
        self.envShapeSet = {}
        for world in self.universeEnvs:
            shapeset = []
            for j in range(len(self.universeEnvs[world])):
                shape_list = arcade.ShapeElementList()

                # --- Create all the rectangles

                # We need a list of all the points and colors
                point_list = []
                color_list = []

                # Now calculate all the points
                for x in range(0, SCREEN_WIDTH, SQUARE_SPACING):
                    for y in range(0, SCREEN_HEIGHT, SQUARE_SPACING):
                        # Calculate where the four points of the rectangle will be if
                        # x and y are the center
                        top_left = (x, y + SQUARE_HEIGHT)
                        top_right = (x + SQUARE_WIDTH, y + SQUARE_HEIGHT)
                        bottom_right = (x + SQUARE_WIDTH, y)
                        bottom_left = (x, y)

                        # Add the points to the points list.
                        # ORDER MATTERS!
                        # Rotate around the rectangle, don't append points caty-corner
                        point_list.append(top_left)
                        point_list.append(top_right)
                        point_list.append(bottom_right)
                        point_list.append(bottom_left)

                        # Add a color for each point. Can be different colors if you want
                        # gradients.
                        for i in range(4):
                            color_list.append(arcade.color.LIGHT_GRAY)

                nzWalls = self.wallMatrixSet[world][j].nonzero()
                for i in range(len(nzWalls[0])):
                    x = nzWalls[1][i] * SQUARE_SPACING
                    y = nzWalls[0][i] * SQUARE_SPACING
                    top_left = (x, y + SQUARE_SPACING)
                    top_right = (x + SQUARE_SPACING, y + SQUARE_SPACING)
                    bottom_right = (x + SQUARE_SPACING, y)
                    bottom_left = (x, y)

                    point_list.append(top_left)
                    point_list.append(top_right)
                    point_list.append(bottom_right)
                    point_list.append(bottom_left)

                    for i in range(4):
                        color_list.append(arcade.color.BLACK)

                shape = arcade.create_rectangles_filled_with_colors(point_list, color_list)
                shape_list.append(shape)
                shapeset.append(shape_list)
            self.envShapeSet[world] = shapeset


    def on_draw(self):
        """
        Render the screen.
        """

        # This command has to happen before we start drawing
        # arcade.start_render()
        #
        # Draw the grid
        # for row in range(ROW_COUNT):
        #     for column in range(COLUMN_COUNT):
        #         # Figure out what color to draw the box
        #         if self.player.x == column and self.player.y == row:
        #             color = self.player.color
        #         elif self.grid[row][column] == 1:
        #             color = arcade.color.GREEN
        #         else:
        #             color = arcade.color.WHITE
        #
        #         # Do the math to figure out where the box is
        #         x = (MARGIN + WIDTH) * column + MARGIN + WIDTH // 2
        #         y = (MARGIN + HEIGHT) * row + MARGIN + HEIGHT // 2
        #
        #         # Draw the box
        #         arcade.draw_rectangle_filled(x, y, WIDTH, HEIGHT, color)


        # This command has to happen before we start drawing
        arcade.start_render()

        # Start timing how long this takes
        draw_start_time = timeit.default_timer()

        # --- Draw all the rectangles
        self.envShapeSet[self.agent.worldIdx][self.currentPeriod].draw()
        # startsquarex = (SQUARE_SPACING * 5 + SQUARE_SPACING // 2)
        # startsquarey = (SQUARE_SPACING * 5 + SQUARE_SPACING // 2)
        startsquarex = (SQUARE_SPACING * STARTX + SQUARE_SPACING // 2)
        startsquarey = (SQUARE_SPACING * STARTY + SQUARE_SPACING // 2)
        arcade.draw_rectangle_filled(startsquarex, startsquarey, SQUARE_WIDTH, SQUARE_HEIGHT, arcade.color.LIGHT_SLATE_GRAY)

        buttonstateNonZero = np.array(self.buttonStateSet[0]).flatten().nonzero()[0]
        buttonstates = np.array(self.buttonStateSet[0]).flatten()
        for i in buttonstateNonZero:
            row, col = self.cEnv.ss.liToCoord[i]
            if buttonstates[i] == 1:
                x = (SQUARE_SPACING * col) + HALF_SQUARE_WIDTH
                y = (SQUARE_SPACING * row) + HALF_SQUARE_HEIGHT
                arcade.draw_rectangle_filled(x, y, SQUARE_WIDTH, SQUARE_HEIGHT, arcade.color.YELLOW)
            if buttonstates[i] == 2:
                x = (SQUARE_SPACING * col) + HALF_SQUARE_WIDTH
                y = (SQUARE_SPACING * row) + HALF_SQUARE_HEIGHT
                arcade.draw_rectangle_filled(x, y, SQUARE_WIDTH, SQUARE_HEIGHT, arcade.color.LIGHT_YELLOW)

        for i, g in enumerate(self.goalStates):
            xcoord = self.cEnv.ss.liToCoord[g][1]
            ycoord = self.cEnv.ss.liToCoord[g][0]
            x = (SQUARE_SPACING * xcoord + SQUARE_SPACING // 2)
            y = (SQUARE_SPACING * ycoord + SQUARE_SPACING // 2)
            if self.currentTime <= self.goalDLmeans[i] and not (g in self.agent.completedGoals):
                sizeFraction = 1-self.currentTime/self.goalDLmeans[i]
                # sizeFraction = 1
                arcade.draw_rectangle_filled(x, y, np.ceil(SQUARE_WIDTH*sizeFraction), np.ceil(SQUARE_HEIGHT*sizeFraction), arcade.color.INDIA_GREEN)
            elif g in self.agent.completedGoals:
                arcade.draw_circle_filled(x, y, np.ceil(0.8*SQUARE_WIDTH//2), arcade.color.INDIA_GREEN)

        bstates = np.setdiff1d(list(self.goalMasterDict.keys()), self.goalStates)
        for b in bstates:
            for i in self.goalMasterDict[b]['walls']:
                if self.wallMatrixSet[self.agent.worldIdx][self.currentPeriod].flatten()[i]==1:
                    row, col = self.cEnv.ss.liToCoord[i]
                    x = (SQUARE_SPACING * col) + HALF_SQUARE_WIDTH
                    y = (SQUARE_SPACING * row) + HALF_SQUARE_HEIGHT
                    arcade.draw_rectangle_filled(x, y, SQUARE_WIDTH, SQUARE_HEIGHT, arcade.color.BROWN)

        x = (SQUARE_SPACING * self.agent.x + SQUARE_SPACING // 2)
        y = (SQUARE_SPACING * self.agent.y + SQUARE_SPACING // 2)
        arcade.draw_rectangle_filled(x, y, HALF_SQUARE_WIDTH, HALF_SQUARE_HEIGHT, self.agent.color)
        # if self.agent.successflag == True:
        #     self.agent.successflag = False
        #     arcade.draw_rectangle_filled(x, y, HALF_SQUARE_WIDTH, HALF_SQUARE_HEIGHT, arcade.color.ORANGE)
        # else:
        #     arcade.draw_rectangle_filled(x, y, HALF_SQUARE_WIDTH, HALF_SQUARE_HEIGHT, self.agent.color)
            # arcade.draw_rectangle_filled(x, y, SQUARE_WIDTH, SQUARE_HEIGHT, arcade.color.LIGHT_BLUE)



        # # Figure out our output
        # output = f"Time: {minutes:02d}:{seconds:02d}"
        #
        # # Output the timer text.
        # arcade.draw_text(output, 300, 300, arcade.color.BLACK, 30)

        # output = f"Drawing time: {self.draw_time:.3f} seconds per frame."
        # arcade.draw_text(output, 20, SCREEN_HEIGHT - 40, arcade.color.WHITE, 18)

        # self.draw_time = timeit.default_timer() - draw_start_time


    def on_mouse_press(self, x, y, button, modifiers):
        """
        Called when the user presses a mouse button.
        """

        # Change the x/y screen coordinates to grid coordinates
        column = x // SQUARE_SPACING
        row = y // SQUARE_SPACING

        print(f"Click coordinates: ({x}, {y}). Grid coordinates: ({row}, {column})")

        # Make sure we are on-grid. It is possible to click in the upper right
        # corner in the margin and go to a grid location that doesn't exist
        if row < ROW_COUNT and column < COLUMN_COUNT:

            # Flip the location between 1 and 0.
            if self.grid[row][column] == 0:
                self.grid[row][column] = 1
            else:
                self.grid[row][column] = 0

    def update(self, dt):
        """ Move everything """
        self.agent.moveForInput()
        if self.spaceFlag or self.initFlag:
        # if self.spaceFlag:
            self.initFlag = True
            self.spaceFlag = False

            # agentLI = self.agent.getLIss()
            # pol = self.agent.currentPolicy
            # self.agent.moveForSolutionDeterministic(self.cEnv.lmdp.bigU[((self.ns * pol) + agentLI), :])


            if self.currentPeriod != self.periodlist[self.currentTime]:
                period = self.periodlist[self.currentTime]
                self.currentPeriod = period
                pi_list = self.universeReachMaps[self.agent.worldIdx]['pi_list']
                self.agent.currentPolicy = pi_list[self.currentPeriod][self.agent.currentGoal][self.agent.getLIss()]
                self.cEnv = self.universeEnvs[self.agent.worldIdx][self.currentPeriod]
                self.agent.currentSS = self.cEnv.ss


            agentLI = self.agent.getLIss()
            pol = self.agent.currentPolicy
            self.agent.moveForSolutionDeterministic(self.cEnv.lmdp.bigU[((self.ns * pol) + agentLI), :])

            if self.agent.getLIss() == self.allgoallist[self.agent.currentGoal]:
                if self.goalMasterDict[self.allgoallist[self.agent.currentGoal]]['type'] == 'button':
                    wallchanges = self.goalMasterDict[self.allgoallist[self.agent.currentGoal]]['walls']
                    newwallstr = matstring(wallTransform(wallchanges, self.cEnv.ss.wallMatrix))
                    newworldidx = self.map2worldIdx[newwallstr]
                    self.agent.worldIdx = newworldidx
                    self.cEnv = self.universeEnvs[self.agent.worldIdx][self.currentPeriod]
                self.agent.setNextGoalAndStitch()
                self.agent.successflag = True
                # self.agent.goalsAchieved[self.agent.plan[self.agent.currentGoal]] = 1
                self.agent.currentPolicy = self.agent.stichSetForPlan[self.agent.planIndex]

            self.currentTime += 1
            self.agent.currentTime += 1

            print('Current Time: ' + str(self.currentTime))
            print('Current Period: ' + str(self.currentPeriod))
            print('Current Policy: ' + str(self.agent.currentPolicy))


    def on_key_press(self, key, modifiers):
        """
        Called whenever a key is pressed.
        """
        if key == arcade.key.UP:
            self.agent.delta_y = MAX_SPEED #min(self.player.delta_y + MOVEMENT_SPEED, MAX_SPEED)
        elif key == arcade.key.DOWN:
            self.agent.delta_y = MIN_SPEED #max(self.player.delta_y - MOVEMENT_SPEED, MIN_SPEED)
        elif key == arcade.key.LEFT:
            self.agent.delta_x = MIN_SPEED #max(self.player.delta_x - MOVEMENT_SPEED, MIN_SPEED)
        elif key == arcade.key.RIGHT:
            self.agent.delta_x = MAX_SPEED #min(self.player.delta_x + MOVEMENT_SPEED, MAX_SPEED)
        elif key == arcade.key.Q:
            arcade.window_commands.close_window()
        elif key == arcade.key.SPACE:
            self.spaceFlag = True

    def on_key_release(self, key, modifiers):
        """
        Called when the user releases a key.
        """
        if key == arcade.key.UP or key == arcade.key.DOWN:
            self.agent.delta_y = 0
        elif key == arcade.key.LEFT or key == arcade.key.RIGHT:
            self.agent.delta_x = 0

    # def on_mouse_drag(self, x: float, y: float, dx: float, dy: float, buttons: int, modifiers: int):

def to_sparse(x):
    """ From https://discuss.pytorch.org/t/how-to-convert-a-dense-matrix-to-a-sparse-one/7809 """
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Author: Philip: https://stackoverflow.com/users/2142071/philip
    From: https://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """
    if not isinstance(mat, csr):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:, mask]
    else:
        return mat

def save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def try_to_load_as_pickled_object_or_None(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    try:
        input_size = os.path.getsize(filepath)
        bytes_in = bytearray(0)
        with open(filepath, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    except:
        return None
    return obj


def plot(data, onezero, *walls):
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    if walls:
        w = walls[0]
        w = np.abs(w.flatten() - 1)
        data = data * w

    temp = data.copy()
    temp[temp > 0.02] = np.max(temp.flatten())

    if data.ndim == 1 and data.size == ROW_COUNT*COLUMN_COUNT:
        data = data.reshape(ROW_COUNT, COLUMN_COUNT)

    norm = plt.Normalize(vmin=np.min(temp), vmax=np.max(temp))
    if onezero:
        plt.imshow(data, vmin=0, vmax=1, origin='lower')
    else:
        plt.imshow(data, norm=norm, origin='lower')
        # plt.imshow(data, aspect='auto', norm=norm, origin='lower')
    # ax.view_init(30, angle)

    plt.show()


def plot_at_y(arr, val, **kwargs):
    plt.plot(arr, np.zeros_like(arr) + val, 'x', **kwargs)
    plt.show()


def polquiver(u, liToCoord, wallList, wallMat, valmap=[]):
    u = u.todense()
    u[u < 0.03] = 0
    X, Y, U, V = [], [], [], []
    for i in range(u.shape[0]):
        if i not in wallList:
            for j in u[i, :].nonzero()[1]:
                xx = liToCoord[i][1]
                yy = liToCoord[i][0]
                uu = (liToCoord[j][1] - xx) * u[i, j]
                vv = (liToCoord[j][0] - yy) * u[i, j]
                X.append(xx)
                Y.append(yy)
                U.append(uu)
                V.append(vv)
    X = np.array(X)
    Y = np.array(Y)
    U = np.array(U)
    V = np.array(V)
    Z = np.array([0]*X.size)
    # plt.hold
    fig = plt.figure()
    # ax = fig.gca(projection='3d')

    xx, yy = np.meshgrid(np.linspace(0, 1, COLUMN_COUNT), np.linspace(0, 1, ROW_COUNT))

    # create vertices for a rotated mesh (3D rotation matrix)
    Xp = xx
    Yp = yy
    Zp = 0 * np.ones(Xp.shape)


    # ax.plot_surface(valmap)

    # ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
    # cset = ax.contourf(X, Y, Z, zdir='z', offset=-100,
    #                    levels=np.linspace(-100, 100, 1200), cmap=plt.cm.jet)
    # cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=plt.cm.jet)
    # cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=plt.cm.jet)
    # ax.set_xlabel('X')
    # ax.set_xlim(-40, 40)
    # ax.set_ylabel('Y')
    # ax.set_ylim(-40, 40)
    # ax.set_zlabel('Z')
    # ax.set_zlim(-100, 100)
    # plt.show()

    # ax = Axes3D(fig)

    # ax2 = fig.add_subplot(111, projection='3d')
    # ax2.plot_surface(Xp, Yp, Zp, rstride=1, cstride=1, facecolors=plt.cm.BrBG(valmap), shade=False)

    # ax1 = fig.add_subplot(121)
    # ax1.imshow(valmap, cmap=plt.cm.BrBG, interpolation='nearest', origin='lower', extent=[0, 1, 0, 1])

    # ax.quiver(X, Y, np.array([1]*X.size), U, V, np.array([0]*X.size), color='r')
    # ax = fig.add_subplot(111, projection='3d')
    # if valmap.ndim == 1 and valmap.size == ROW_COUNT*COLUMN_COUNT:
    #     data = valmap.reshape(ROW_COUNT, COLUMN_COUNT)
    # plt.imshow(valmap, aspect='auto', origin='lower')
    # if valmap.any():
    #     plot(valmap)

    w = np.abs(wallMat.flatten() - 1)

    plt.quiver(X, Y, U, V, width=0.006, scale=18, headwidth=3, headlength=3, headaxislength=3, color='r')
    plt.imshow(np.abs(wallMat-1)*.85, vmin=0, vmax=1, origin='lower', cmap='gray')
    plt.show()


def onehot(inds, size):
    oh = np.zeros(size)
    oh[inds] = 1
    return oh

def addGoalDlsToPeriodList():
    wallPeriodStartList = np.array([0] + list(timeStepArrayWalls[0:-1:1])).cumsum()
    combinedPeriodStartList = wallPeriodStartList.copy()
    sdiff = np.sort(np.setdiff1d(goalDLmeans, wallPeriodStartList))
    periodToEnvMap = np.arange(len(wallPeriodStartList))
    # combinedPeriodMap = wallPeriodMap.copy()
    # for i, e in enumerate(sdiff):
    #     a = wallPeriodStartList[wallPeriodStartList < e][-1]
    #     idx = np.where(combinedPeriodStartList==a)[0][0]
    #     periodToEnvMap = np.insert(periodToEnvMap, idx+1, periodToEnvMap[idx])
    #     combinedPeriodStartList = np.insert(combinedPeriodStartList, idx+1, e)
    # combinedPeriodStartList = np.sort(combinedPeriodStartList)
    timeStepArrayWithGoals = np.diff(combinedPeriodStartList)

    return []

def powerset(iterable):
    "list(powerset([1,2,3])) --> [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def pset(myset):
  if not myset: # Empty list -> empty set
    return [set()]

  r = []
  for y in myset:
    sy = set((y,))
    for x in pset(myset - sy):
      if x not in r:
        r.extend([x, x|sy])
  return r

def matstring(mat):
    return " ".join(str(x) for x in mat.flatten())

def wallTransform(buttonwallinds, curwallmat):
    buttonwallmat = np.abs(onehot(buttonwallinds, COLUMN_COUNT*ROW_COUNT)-1)
    newwallmat = np.reshape((np.logical_and(curwallmat.flatten(), buttonwallmat) + 0), [ROW_COUNT, COLUMN_COUNT])
    return newwallmat

def main():

    # UTILITIES
    loadfile = 0
    problemStructureFile = 'savedProblemSimple.obj'
    preloadtemplate = 0
    templateFile = 'wallTemplate_3by3.obj'
    saveEnvData = 0
    loadEnvData = 0
    loadenvfilename = 'envDataButton11.obj'
    saveenvfilename = 'envDataButton12.obj'
    shuffleEnvs = 0
    ns = ROW_COUNT*COLUMN_COUNT

    # LOAD SAVED PROJECT
    if os.path.isfile(problemStructureFile) and loadfile:
        filehandler = open(problemStructureFile, 'rb')
        params = pickle.load(filehandler)
        filehandler.close()

        params['loadenvfilename'] = loadenvfilename if loadEnvData else None
        params['saveEnvData'] = saveEnvData
        params['loadEnvData'] = loadEnvData

        game = MyGame(**params)
        game.setup()
        arcade.run()

    # CREATE NEW PROJECT
    else:
        if preloadtemplate and os.path.isfile(templateFile):
            filehandler = open(templateFile, 'rb')
            wallTemplate = pickle.load(filehandler)
            initGrid = wallTemplate['wt']
        else:
            initGrid = np.zeros([ROW_COUNT, COLUMN_COUNT])

        wallmatset = []
        # timeStepArray = np.array([4, 12, 4])
        # timeStepArrayWalls = np.array([5, 16, 18, 50])  # Last time should be arbitrarily long

        # timeStepArrayWalls = np.array([9, 5, 8, 30])  # Forward
        timeStepArrayWalls = np.array([8, 5, 9, 30])  # Reverse

        goalstates = set()
        buttonstates = np.zeros([ROW_COUNT, COLUMN_COUNT])
        buttonstatelist = []
        buttonToWall = defaultdict(list)
        for i in range(len(timeStepArrayWalls)):
            window = Setup(SCREEN_WIDTH, SCREEN_HEIGHT, goalstates, buttonstates, initGrid, buttonToWall)
            window.setup()
            arcade.run()
            wallmatset.append(window.grid.copy())
            buttonToWall = window.buttonToWall
            buttonstatelist.append(window.buttonStates.copy())
            goalstates = window.goalStates
            buttonstates = window.buttonStates
            initGrid = window.grid.copy()
            del window

        # Save first environment as template
        # wallTemplateSave = {'wt': wallmatset[0]}
        # filehandler = open('wallTemplate_3by3.obj', 'wb')
        # pickle.dump(wallTemplateSave, filehandler)
        # filehandler.close()

        goalstates = np.sort(list(goalstates))
        del buttonToWall[None]

        buttonIdxs = list(buttonToWall.keys())

        maingoalDLmeans = [18, 18, 24, 24]
        maingoalDLvars = [0.2] * len(maingoalDLmeans)

        buttonDLmeans = [np.max(maingoalDLmeans)]*len(buttonIdxs)
        buttonDLvars = [0.2]*len(buttonIdxs)

        allgoalDLmeans = maingoalDLmeans + [np.max(maingoalDLmeans)] * len(list(buttonToWall.keys()))
        allgoalDLvars = maingoalDLvars + [0.2] * len(list(buttonToWall.keys()))

        allgoallist = list(goalstates) + list(buttonToWall.keys())
        goalMaster = [{'idx': e, 'mean': maingoalDLmeans[i], 'var': maingoalDLvars[i], 'type': 'primary'} for i, e in enumerate(goalstates)]
        buttonGoals = [{'idx': e, 'mean': np.max(maingoalDLmeans), 'var': buttonDLvars[i], 'type': 'button', 'walls': buttonToWall[e]} for i, e in enumerate(buttonIdxs)]
        for d in buttonGoals:
            goalMaster.append(d)

        goalMasterDict = dict(zip(allgoallist, goalMaster))

        wallPeriodStartList = np.array([0]+list(timeStepArrayWalls[0:-1:1])).cumsum()
        periodToEnvMap = np.arange(len(wallPeriodStartList))
        periodToEnvMap = np.array(list(periodToEnvMap) + [periodToEnvMap[-1]])
        # combinedPeriodStartList = wallPeriodStartList.copy()
        # sdiff = np.sort(np.setdiff1d(goalDLmeans, wallPeriodStartList))
        # combinedPeriodMap = wallPeriodMap.copy()
        # for i, e in enumerate(sdiff):
        #     a = wallPeriodStartList[wallPeriodStartList < e][-1]
        #     idx = np.where(combinedPeriodStartList==a)[0][0]
        #     periodToEnvMap = np.insert(periodToEnvMap, idx+1, periodToEnvMap[idx])
        #     combinedPeriodStartList = np.insert(combinedPeriodStartList, idx+1, e)
        # combinedPeriodStartList = np.sort(combinedPeriodStartList)
        # timeStepArrayWithGoals = np.diff(combinedPeriodStartList)

        # universe_walls = {'main': wallmatset}
        universe_walls = {0: wallmatset}
        buttonPowSet = powerset(buttonIdxs)
        map2worldIdx = {}  # This function maps a sringified representation of the wallmatrix to the world index
        del buttonPowSet[0]
        for i, bset in enumerate(buttonPowSet):
            i = i+1
            newwms = wallmatset.copy()
            for d in bset:
                walls = buttonToWall[d]
                buttonwallmat = np.abs(onehot(walls, ns)-1)
                newwms = [np.reshape((np.logical_and(wallmat.flatten(), buttonwallmat)+0), [ROW_COUNT, COLUMN_COUNT]) for wallmat in newwms]

            for newmat in newwms:
                map2worldIdx[matstring(newmat)] = i

            universe_walls[i] = newwms

        intrinsicDLmean = 35
        intrinsicDLvar = 0.6
        gameParamDict = {
            'width': SCREEN_WIDTH,
            'height': SCREEN_HEIGHT,
            'wallmatset': universe_walls,
            'timeStepArray': timeStepArrayWalls,
            'goalstates': goalstates,
            'goalDLmeans': allgoalDLmeans,
            'goalDLvars': allgoalDLvars,
            'intrinsicDLmean': intrinsicDLmean,
            'intrinsicDLvar': intrinsicDLvar,
            'periodToEnvMap': periodToEnvMap,
            'buttonstatelist': buttonstatelist,
            'allgoallist': allgoallist,
            'buttonPowSet': buttonPowSet,
            'map2worldIdx': map2worldIdx,
            'goalmasterdict': goalMasterDict,
            'loadenvfilename': '',
            'loadEnvData': False,
            'saveEnvData': True,
            'saveenvfilename': saveenvfilename}

        filehandler = open('savedProblemSimple.obj', 'wb')
        pickle.dump(gameParamDict, filehandler)
        filehandler.close()
        game = MyGame(**gameParamDict)
        # gameWindow = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, wallmatset, timeStepArray, goalstates, dl, buttonstatelist, None)

        game.setup()
        arcade.run()


if __name__ == "__main__":
    main()



# Random Plotting tools
    # X, Y = np.meshgrid(np.arange(COLUMN_COUNT), np.arange(ROW_COUNT))
    # Xf = X.flatten() + 0.5
    # Yf = Y.flatten() + 0.5
    # endpnts = np.array([self.cEnv.ss.liToCoord[i] for i in self.piSet[3]])

    # for k, _ in enumerate(self.piSet):
    #     print(k)
    #     endpnts = np.array([self.cEnv.ss.liToCoord[j] for j in self.piSet[k]])
    #     xarrowlength = endpnts[:, 1] - X.flatten()
    #     yarrowlength = endpnts[:, 0] - Y.flatten()
    #     plt.figure()
    #     plt.title('Agent Dynamics, Period 0')
    #     plt.ylim(0, ROW_COUNT + 1)
    #     plt.xlim(0, COLUMN_COUNT + 1)
    #     for i, a in enumerate(Xf):
    #         plt.arrow(Xf[i], Yf[i], xarrowlength[i], yarrowlength[i], head_width=0.15, length_includes_head=False, head_length=0.1, fc='k', ec='k')
    #     plt.show()
    # plt.quiver(X, Y, endpnts[:, 1], endpnts[:, 0])
    # plt.show()