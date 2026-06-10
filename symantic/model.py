
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:22:50 2023

@author: muthyala.7
"""


from .feature_expansion import nondimensional as fcc

from .feature_expansion import dimensional as dfcc

from .results import FitResult

from .validation import validate_dataframe, validate_operators, validate_dimensions, validate_regularization

import sys

import time

import torch

import numpy as np

import pandas as pd

from sympy import symbols

import matplotlib.pyplot as plt

import matplotlib

class SymanticModel:

  def __init__(self,df,operators=None,multi_task = None,n_expansion=None,n_term=None,sis_features=20,device=None,relational_units = None,initial_screening = None,dimensionality=None,output_dim = None,metrics=[0.06,0.995],disp=False,pareto=False,max_features=None,regularization='l0',reg_alpha=None,l1_ratio=0.5,reg_threshold=1e-4,n_alphas=100,level_pruning=False):
    """Initialize SymanticModel.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. First column is the target, rest are features.
    operators : list of str
        Mathematical operators for feature expansion (e.g. ['+', '*', 'exp']).
    multi_task : tuple or None
        (target_indices, feature_indices) for multi-task regression.
    n_expansion : int or None
        Number of expansion levels. None enables auto-depth mode.
        Note: uses range(1, n_expansion), so n_expansion=2 gives 1 level.
    n_term : int or None
        Max terms per equation. Default 3.
    sis_features : int
        Number of features to keep via SIS screening. Default 20.
    device : str or None
        'cpu', 'cuda', or None (auto-detect). Default None selects
        'cuda' when a GPU is available, otherwise 'cpu'.
    relational_units : list or None
        Unit relationships for dimensional regression.
    initial_screening : tuple or None
        (n_features, quantile) for initial feature screening.
    dimensionality : list or None
        Sympy dimension expressions for dimensional regression.
    output_dim : sympy expr or None
        Dimension of the target variable.
    metrics : list
        [rmse_threshold, r2_threshold] for auto-depth convergence.
    disp : bool
        Print progress information. Default False.
    pareto : bool
        Compute Pareto front. Default False.
    max_features : int or None
        Maximum features before stopping expansion in auto-depth mode.
        Default: 2000 for non-dimensional, 10000 for dimensional.
        Increase to search for more complex equations (at higher memory cost).
    regularization : str
        Regression method: 'l0' (default, exhaustive combinatorial), 'l1'
        (Lasso), 'l2' (Ridge), or 'elastic_net'. L1/ElasticNet are much
        faster for n_term >= 3 because they avoid enumerating all C(k,n)
        feature combinations.
    reg_alpha : float or None
        Regularization strength. None = auto-select via regularization path.
    l1_ratio : float
        L1/L2 mixing for elastic_net (1.0 = pure L1). Default 0.5.
    reg_threshold : float
        For L2: zero out coefficients below this fraction of max |coef|.
    n_alphas : int
        Number of alpha values in the regularization path. Default 100.
    level_pruning : bool
        When True, prune features between auto-depth expansion levels.
        After regression at each level, keeps only the top `sis_features`
        most correlated derived features (plus all original base features)
        before expanding further. Caps memory growth for deep expansions.
        Default False.
    """
    # Validate inputs
    validate_dataframe(df)
    if operators is not None:
        validate_operators(operators)
    if dimensionality is not None:
        validate_dimensions(dimensionality, df)
    validate_regularization(regularization, reg_alpha, l1_ratio)

    self.operators = operators

    self.regularization = regularization
    self.reg_alpha = reg_alpha
    self.l1_ratio = l1_ratio
    self.reg_threshold = reg_threshold
    self.n_alphas = n_alphas
    self._reg_kwargs = dict(
        regularization=regularization, reg_alpha=reg_alpha,
        l1_ratio=l1_ratio, reg_threshold=reg_threshold, n_alphas=n_alphas,
    )
    self.level_pruning = level_pruning

    self.df=df

    self.no_of_operators = n_expansion

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.device = device

    if n_term == None: self.dimension = 3

    else: self.dimension = n_term

    if sis_features == None: self.sis_features = 10

    else: self.sis_features = sis_features

    self.relational_units = relational_units

    self.initial_screening = initial_screening

    self.dimensionality = dimensionality

    self.output_dim = output_dim


    self.metrics   = metrics

    self.multi_task = multi_task

    self.disp=disp

    self.pareto=pareto

    # Set max_features with sensible defaults per mode
    if max_features is not None:
        self.max_features = max_features
    elif dimensionality is not None:
        self.max_features = 10000
    else:
        self.max_features = 2000

    if multi_task!=None:

        self.multi_task_target = multi_task[0]

        self.multi_task_features = multi_task[1]
    self.final_df = None
    

  def combine_equation(self,row):
      
      terms = row['Equations']
      
      coeffs = row['Coefficients']
      
      intercept = row['Intercepts']
      
      if isinstance(terms, str):
      
          terms = [terms]
      
      equation_parts = []

      for term, coeff in zip(terms, coeffs):
          if pd.isna(coeff):
              continue
          if coeff == 1:
              equation_parts.append(term)
          elif coeff == -1:
              equation_parts.append(f"-{term}")
          else:
              equation_parts.append(f"{coeff:.20f}*{term}")

      equation = " + ".join(equation_parts)

      if intercept != 0:

          if intercept > 0:

              equation = f"{equation} + {intercept:.20f}"

          else:

              equation = f"{equation} - {abs(intercept):.20f}"
      
      return equation   

    
  def _build_pareto_result(self, final):
      """Post-process auto-depth Pareto front into FitResult."""
      self.final_df = final

      # Safely normalize loss and complexity to [0, 1]
      l_min, l_max = final['Loss'].min(), final['Loss'].max()
      final['Normalized_Loss'] = (final['Loss'] - l_min) / (l_max - l_min) if l_max > l_min else 0.0

      c_min, c_max = final['Complexity'].min(), final['Complexity'].max()
      final['Normalized_Complexity'] = (final['Complexity'] - c_min) / (c_max - c_min) if c_max > c_min else 0.0

      final['Distance_to_Utopia'] = np.sqrt(final['Normalized_Loss']**2 + final['Normalized_Complexity']**2)

      final_edited = pd.DataFrame()
      final_edited['Loss'] = final['Loss']
      final_edited['Complexity'] = final['Complexity']
      final_edited['R2'] = final['Score']
      final_edited['Equation'] = final.apply(self.combine_equation, axis=1)

      # 1. Prioritize solutions that meet convergence targets if any do
      meets_metrics = (final_edited['Loss'] <= self.metrics[0]) & (final_edited['R2'] >= self.metrics[1])
      if meets_metrics.any():
          # Pick the simplest model among those that meet the metrics
          utopia_row = final_edited[meets_metrics]['Complexity'].idxmin()
      else:
          # 2. Otherwise fall back to "Distance to Utopia" in normalized space
          # Handle potential ties by picking the one with better R2
          min_dist = final['Distance_to_Utopia'].min()
          is_min = final['Distance_to_Utopia'] == min_dist
          if is_min.sum() > 1:
              utopia_row = final_edited[is_min]['R2'].idxmax()
          else:
              utopia_row = final['Distance_to_Utopia'].idxmin()

      if self.disp:
          print('Pareto set generated. Access via result.pareto_front')

      return FitResult(
          rmse=float(final_edited.Loss[utopia_row]),
          equation=str(final_edited.Equation[utopia_row]),
          r2=float(final_edited.R2[utopia_row]),
          complexity=float(final_edited.Complexity[utopia_row]),
          pareto_front=final_edited,
      )

  def fit(self):
      """Run symbolic regression and return a FitResult.

      Returns
      -------
      FitResult
          Unified result object. Supports backward-compatible tuple unpacking:
          - Auto-depth: ``res, pareto_df = model.fit()``
          - Fixed-depth: ``rmse, equation, r2 = model.fit()``
          - Multi-task: ``rmse, equation, r2, equations = model.fit()``
      """
      if self.dimensionality == None:

        if self.operators==None: sys.exit('Please provide the operators set for the non dimensional Regression!!')

        if self.multi_task!=None:

            if self.disp: print('************************************* Performing MultiTask Symbolic regression!!..**************************************************************** \n')

            equations =[]

            for i in range(len(self.multi_task_target)):

                if self.disp: print('***************************************** Performing symbolic regression of',i+1,'Target variables******************************************** \n')

                list1 =[]
                list1.extend([self.multi_task_target[i]]+self.multi_task_features[i])
                df1 = self.df.iloc[:,list1]

                if self.no_of_operators==None:

                    st = time.time()
                    rmse,equation,r2,_ = fcc.feature_space_construction(self.operators,df1,self.no_of_operators,self.device,self.initial_screening,self.metrics,dimension=self.dimension,sis_features=self.sis_features,disp=self.disp,pareto=self.pareto,max_features=self.max_features,level_pruning=self.level_pruning,**self._reg_kwargs).feature_space()
                    if self.disp: print('************************************************ Autodepth regression completed in::', time.time()-st,'seconds ************************************************ \n')

                    equations.append(equation)
                    if i+1 == len(self.multi_task_target):
                        if self.disp: print('Equations found::',equations)
                        return FitResult(rmse=rmse, equation=equation, r2=r2, all_equations=equations)
                    else:continue

                else:

                    x,y,names,complexity = fcc.feature_space_construction(self.operators,df1,self.no_of_operators,self.device,self.initial_screening,disp=self.disp,pareto=self.pareto,max_features=self.max_features).feature_space()
                    from .regression.factory import get_regressor
                    _Reg = get_regressor(self.regularization, dimensional=False)
                    rmse, equation,r2,r,c,n,intercepts,coeffs,_ =  _Reg(x,y,names,complexity,self.dimension,self.sis_features,self.device,**self._reg_kwargs).regressor_fit()

                    equations.append(equation)
                    if i+1 == len(self.multi_task_target):
                        if self.disp: print('Equations found::',equations)
                        return FitResult(rmse=rmse, equation=equation, r2=r2, all_equations=equations)
                    else: continue

        elif self.no_of_operators==None:

            st = time.time()
            rmse,equation,r2,final = fcc.feature_space_construction(self.operators,self.df,self.no_of_operators,self.device,self.initial_screening,self.metrics,dimension=self.dimension,sis_features=self.sis_features,disp=self.disp,pareto=self.pareto,max_features=self.max_features,level_pruning=self.level_pruning,**self._reg_kwargs).feature_space()
            if self.disp: print('************************************************ Autodepth regression completed in::', time.time()-st,'seconds ************************************************ \n')

            return self._build_pareto_result(final)

        else:

            x,y,names,complexity = fcc.feature_space_construction(self.operators,self.df,self.no_of_operators,self.device,self.initial_screening,disp=self.disp,max_features=self.max_features).feature_space()
            from .regression.factory import get_regressor
            _Reg = get_regressor(self.regularization, dimensional=False)
            rmse, equation,r2,r,c,n,intercepts,coeffs,_ =  _Reg(x,y,names,complexity,self.dimension,self.sis_features,self.device,**self._reg_kwargs).regressor_fit()

            return FitResult(rmse=rmse, equation=equation, r2=r2)

      else:

        if self.multi_task!=None:

            if self.disp: print('************************************************ Performing MultiTask Symbolic regression!!..************************************************ \n')

            equations =[]

            for i in range(len(self.multi_task_target)):

                if self.disp: print('************************************************ Performing symbolic regression of',i+1,'Target variables....************************************************ \n')

                list1 =[]
                list1.extend([self.multi_task_target[i]]+self.multi_task_features[i])
                df1 = self.df.iloc[:,list1]

                if self.no_of_operators==None:

                    st = time.time()
                    rmse,equation,r2,final = dfcc.feature_space_construction(df1,self.operators,self.relational_units,self.initial_screening,self.no_of_operators,self.device,self.dimensionality,self.metrics,self.output_dim,disp=self.disp,pareto=self.pareto,max_features=self.max_features,level_pruning=self.level_pruning,**self._reg_kwargs).feature_expansion()
                    if self.disp: print('************************************************ Autodepth regression completed in::', time.time()-st,'seconds ************************************************ \n')

                    equations.append(equation)
                    if i+1 == len(self.multi_task_target):
                        if self.disp: print('Equations found::',equations)
                        return FitResult(rmse=rmse, equation=equation, r2=r2, all_equations=equations)
                    else:continue

                else:

                    x,y,names,dim,complexity = dfcc.feature_space_construction(df1,self.operators,self.relational_units,self.initial_screening,self.no_of_operators,self.device,self.dimensionality,disp=self.disp,pareto=self.pareto,max_features=self.max_features).feature_expansion()
                    from .regression.factory import get_regressor
                    _Reg = get_regressor(self.regularization, dimensional=True)
                    rmse,equation,r2,_,_,_,_,_,_ = _Reg(x,y,names,dim,complexity,self.dimension,self.sis_features,self.device,self.output_dim,disp=self.disp,pareto=self.pareto,**self._reg_kwargs).regressor_fit()

                    equations.append(equation)
                    if i+1 == len(self.multi_task_target):
                        if self.disp: print('Equations found::',equations)
                        return FitResult(rmse=rmse, equation=equation, r2=r2, all_equations=equations)
                    else: continue

        if self.no_of_operators==None:

            st = time.time()
            rmse,equation,r2,final = dfcc.feature_space_construction(self.df,self.operators,self.relational_units,self.initial_screening,self.no_of_operators,self.device,self.dimensionality,self.metrics,self.output_dim,disp=self.disp,pareto=self.pareto,max_features=self.max_features,level_pruning=self.level_pruning,**self._reg_kwargs).feature_expansion()
            if self.disp: print('************************************************ Autodepth regression completed in::', time.time()-st,'seconds ************************************************ \n')

            return self._build_pareto_result(final)

        else:

            x,y,names,dim,complexity = dfcc.feature_space_construction(self.df,self.operators,self.relational_units,self.initial_screening,self.no_of_operators,self.device,self.dimensionality,disp=self.disp,pareto=self.pareto,max_features=self.max_features).feature_expansion()
            from .regression.factory import get_regressor
            _Reg = get_regressor(self.regularization, dimensional=True)
            rmse,equation,r2,_,_,_,_,_,_ = _Reg(x,y,names,dim,complexity,self.dimension,self.sis_features,self.device,self.output_dim,disp=self.disp,pareto=self.pareto,**self._reg_kwargs).regressor_fit()

            return FitResult(rmse=rmse, equation=equation, r2=r2)
        
  def plot_pareto_front(self):
    
    import matplotlib.pyplot as plt

    import matplotlib  
    plt.figure(figsize=(10, 8))
    
    plt.scatter(self.final_df['Complexity'],self.final_df['Loss'], c='red', label='Pareto front')
    
    plt.step(self.final_df['Complexity'],self.final_df['Loss'], 'r-', where='post', label='Pareto Line')
    
    plt.scatter(self.final_df['Complexity'].min(),self.final_df['Loss'].min(), c='green', label='Utopia',marker='*',s=100)
    
    plt.xlabel(r'Complexity = $k \log n$ (bits)',weight='bold') 
    
    plt.ylabel('Accuracy (RMSE)',weight='bold')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True)
    
    plt.title('Pareto Frontier')
    
    plt.show()

  def evaluate(self, equation, df_test, custom_functions=None):
    """Evaluate a symbolic equation on new data.

    Parameters
    ----------
    equation : str
        Equation string as returned by ``fit()``.
    df_test : pd.DataFrame
        Test data whose column names match the variable names in *equation*.
    custom_functions : dict or None
        Mapping of ``{name: callable}`` for any user-defined functions
        appearing in the equation.

    Returns
    -------
    predictions : pd.Series or None
        Predicted values, or ``None`` if evaluation fails.
    equation : str
        The (numpy-substituted) equation that was evaluated.
    """
    import re

    # Build a local namespace with column data + numpy helpers
    local_ns = {col: df_test[col] for col in df_test.columns}
    local_ns['np'] = np

    if custom_functions:
        local_ns.update(custom_functions)

    # Convert ^ to ** for Python exponentiation
    equation = re.sub(r'\^', '**', equation)

    # Substitute symbolic function names with numpy equivalents
    equation = re.sub(r'\bexp\b', 'np.exp', equation)
    equation = re.sub(r'\bcos\b', 'np.cos', equation)
    equation = re.sub(r'\bsin\b', 'np.sin', equation)
    equation = re.sub(r'\btan\b', 'np.tan', equation)
    equation = re.sub(r'\bcsc\b', '1/np.sin', equation)
    equation = re.sub(r'\bsec\b', '1/np.cos', equation)
    equation = re.sub(r'\bcot\b', '1/np.tan', equation)

    equation = re.sub(r'\basin\b', 'np.arcsin', equation)
    equation = re.sub(r'\bacos\b', 'np.arccos', equation)
    equation = re.sub(r'\batan\b', 'np.arctan', equation)
    equation = re.sub(r'\bacsc\b', '1/np.arcsin', equation)
    equation = re.sub(r'\basec\b', '1/np.arccos', equation)
    equation = re.sub(r'\bacot\b', '1/np.arctan', equation)

    equation = re.sub(r'\bsinh\b', 'np.sinh', equation)
    equation = re.sub(r'\bcosh\b', 'np.cosh', equation)
    equation = re.sub(r'\btanh\b', 'np.tanh', equation)
    equation = re.sub(r'\bcsch\b', '1/np.sinh', equation)
    equation = re.sub(r'\bsech\b', '1/np.cosh', equation)
    equation = re.sub(r'\bcoth\b', '1/np.tanh', equation)

    equation = re.sub(r'\basinh\b', 'np.arcsinh', equation)
    equation = re.sub(r'\bacosh\b', 'np.arccosh', equation)
    equation = re.sub(r'\batanh\b', 'np.arctanh', equation)

    equation = re.sub(r'\babs\b', 'np.abs', equation)
    equation = re.sub(r'\blog\b', 'np.log10', equation)
    equation = re.sub(r'\bln\b', 'np.log', equation)

    try:
        p = eval(equation, {"__builtins__": {}}, local_ns)
    except Exception as e:
        print(f"Error evaluating equation: {e}")
        return None, equation

    return p, equation
