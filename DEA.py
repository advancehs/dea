# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, maximize, Constraint
import numpy as np
import pandas as pd
from .constant import CET_ADDI, ORIENT_IO, ORIENT_OO, RTS_VRS, RTS_CRS, OPT_DEFAULT, OPT_LOCAL
from .utils import tools


class DEA:
    """Data Envelopment Analysis (DEA)
    """

    def __init__(self, y, x, orient, rts, yref=None, xref=None):
        """DEA: Envelopment problem

        Args:
            y (float): output variable.
            x (float): input variables.
            orient (String): ORIENT_IO (input orientation) or ORIENT_OO (output orientation)
            rts (String): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale)
            yref (String, optional): reference output. Defaults to None.
            xref (String, optional): reference inputs. Defaults to None.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        # Initialize DEA model
        self.__model__ = ConcreteModel()

        self.y, self.x = tools.assert_valid_mupltiple_y_data(y, x)
        self.orient = orient
        self.rts = rts

        if type(yref) != type(None):
            self.yref, self.xref = tools.assert_valid_reference_data(self.yref, self.xref)
        else:
            self.yref, self.xref = self.y, self.x

        I0 = 0  ## 当前被评价决策单元的序号 self.x[I0]
        # Initialize sets
        self.__model__.I = Set(initialize=range(self.x.shape[0]))          ## I 是 被评价决策单元的数量
        self.__model__.I2 = Set(initialize=range(self.refx.shape[0]))      ## I2 是 参考决策单元的数量
        self.__model__.K = Set(initialize=range(self.x.shape[1]))          ## K 是投入个数
        self.__model__.L = Set(initialize=range(len(self.y[0])))           ## L 是产出个数 被评价单元和参考单元的K，L一样

        # Initialize variable
        self.__model__.theta = Var(Set(initialize=range(self.x[I0].shape[0])), doc='efficiency')
        self.__model__.lamda = Var(self.__model__.I, bounds=(0.0, None), doc='intensity variables')

        # Setup the objective function and constraints

        self.__model__.objective = Objective(
            rule=self.__objective_rule(), sense=maximize, doc='objective function')
        self.__model__.input = Constraint(
            self.__model__.K, rule=self.__input_rule(), doc='input constraint')
        self.__model__.output = Constraint(
            self.__model__.L, rule=self.__output_rule(), doc='output constraint')


        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

    def __objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return sum(model.theta)

        return objective_rule

    def __input_rule(self):
        """Return the proper input constraint"""

        def input_rule(model, k):
            return sum(model.lamda[r] * self.xref[r][k] for r in model.R) <= self.x[I0][k]

        return input_rule

    def __output_rule(self):
        """Return the proper output constraint"""

        def output_rule(model, l):
            return sum(model.lamda[r] * self.yref[r][l] for r in model.R) >=model.theta * self.y[I0][l]

        return output_rule

    def __vrs_rule(self):
        def vrs_rule(model, o):
            return sum(model.lamda[o, r] for r in model.R) == 1

        return vrs_rule

    def optimize(self, email=OPT_LOCAL, solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            email (string): The email address for remote optimization. It will optimize locally if OPT_LOCAL is given.
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization
        self.problem_status, self.optimization_status = tools.optimize_model(
            self.__model__, email, CET_ADDI, solver)

    def display_status(self):
        """Display the status of problem"""
        print(self.optimization_status)

    def display_theta(self):
        """Display theta value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.theta.display()

    def display_lamda(self):
        """Display lamda value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.lamda.display()

    def get_status(self):
        """Return status"""
        return self.optimization_status

    def get_theta(self):
        """Return theta value by array"""
        tools.assert_optimized(self.optimization_status)
        theta = list(self.__model__.theta[:].value)
        return np.asarray(theta)

    def get_lamda(self):
        """Return lamda value by array"""
        tools.assert_optimized(self.optimization_status)
        lamda = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.lamda),
                                                           list(self.__model__.lamda[:, :].value))])
        lamda = pd.DataFrame(lamda, columns=['Name', 'Key', 'Value'])
        lamda = lamda.pivot(index='Name', columns='Key', values='Value')
        return lamda.to_numpy()

