import copy
from enum import Enum
from env_v3 import EnvironmentManager, SymbolResult
from func_v3 import FunctionManager, FuncInfo
from intbase import InterpreterBase, ErrorType
from tokenize import Tokenizer

# for v3 spec: object nesting via multiple dots need not supported

class Type(Enum):
  INT = 1
  BOOL = 2
  STRING = 3
  VOID = 4
  FUNC = 5
  OBJECT = 6

class Value:
  def __init__(self, type, value = None):
    self.t = type
    self.v = value

  def value(self):
    return self.v
  
  def set(self, other):
    self.t = other.t 
    self.v = other.v

  def type(self):
    return self.t

# named function closures have no environment since there's no capturing
class NamedFunctionClosure:
  def __init__(self, name):
    self.name = name
  
  def get_name(self):
    return self.name
  
  def get_env(self):
    return None

# lambda closures will have an environment specified for capture purposes
# capture will occur by value
class LambdaClosure:
  def __init__(self, line_num, env):
    super().__init__()
    self.line_num = line_num 
    # make a deep copy of our current function's environment to capture everything by value
    self.env = copy.deepcopy(env)
  
  def get_name(self):
    return FunctionManager.create_lambda_name(self.line_num)
  
  def get_env(self):
    return self.env

class Interpreter(InterpreterBase):
  def __init__(self, console_output=True, input=None, trace_output=False):
    super().__init__(console_output, input)
    self._setup_operations()
    self._setup_default_values()
    self.trace_output = trace_output

  def run(self, program):
    self.program = program
    self._compute_indentation(program)  # determine indentation of every line
    self.tokenized_program = Tokenizer.tokenize_program(program)
    self.func_manager = FunctionManager(self.tokenized_program)
    self.ip = self.func_manager.get_function_info(InterpreterBase.MAIN_FUNC).start_ip
    self.return_stack = []
    self.terminate = False
    self.env_manager = EnvironmentManager()

    while not self.terminate:
      self._process_line()

  def _process_line(self):
    if self.trace_output:
      print(f"{self.ip:04}: {self.program[self.ip].rstrip()}")
    tokens = self.tokenized_program[self.ip]
    if not tokens:
      self._blank_line()
      return

    args = tokens[1:] 

    match tokens[0]:
      case InterpreterBase.ASSIGN_DEF:
        self._assign(args)
      case InterpreterBase.FUNCCALL_DEF:
        self._funccall(args)
      case InterpreterBase.ENDFUNC_DEF:
        self._endfunc()
      case InterpreterBase.IF_DEF:
        self._if(args)
      case InterpreterBase.ELSE_DEF:
        self._else()
      case InterpreterBase.ENDIF_DEF:
        self._endif()
      case InterpreterBase.RETURN_DEF:
        self._return(args)
      case InterpreterBase.WHILE_DEF:
        self._while(args)
      case InterpreterBase.ENDWHILE_DEF:
        self._endwhile(args)
      case InterpreterBase.VAR_DEF: # v2 statements
        self._define_var(args)
      case InterpreterBase.LAMBDA_DEF: # v3 statements
        self._lambda_def(args)
      case InterpreterBase.ENDLAMBDA_DEF: # v3 statements
        self._endlambda()
      case default:
        raise Exception(f'Unknown command: {tokens[0]}')

  def _blank_line(self):
    self._advance_to_next_statement()

  def _assign(self, tokens):
   if len(tokens) < 2: 
     super().error(ErrorType.SYNTAX_ERROR,"Invalid assignment statement")
   vname = tokens[0]
   value_type = self._eval_expression(tokens[1:])

   is_object = False
   if '.' in vname:
     is_object = True
     obj_member = vname.split('.')
     vname = obj_member[0]
     member = obj_member[1]

   existing_value_type = self._get_value(vname)
   if not is_object:
     if existing_value_type.type() != value_type.type():
       super().error(ErrorType.TYPE_ERROR,
                     f"Trying to assign a variable of {existing_value_type.type()} to a value of {value_type.type()}",
                     self.ip)
     self._set_value(vname, value_type)
   else:
     if existing_value_type.type() != Type.OBJECT:
       super().error(ErrorType.TYPE_ERROR, f"Variable {vname} is not an object", self.ip)
     # if it's an object, just re-assign the member in the dictionary to the value of the expression
     obj_dict = existing_value_type.value()
     # Copy value when assigning primitive type into object field
     if value_type.type == Type.OBJECT:
       obj_dict[member] = value_type
     else:
       obj_dict[member] = copy.copy(value_type)

   self._advance_to_next_statement()
  
  def _funccall(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Missing function name to call", self.ip)
    if args[0] == InterpreterBase.PRINT_DEF:
      self._print(args[1:])
      self._advance_to_next_statement()
    elif args[0] == InterpreterBase.INPUT_DEF:
      self._input(args[1:])
      self._advance_to_next_statement()
    elif args[0] == InterpreterBase.STRTOINT_DEF:
      self._strtoint(args[1:])
      self._advance_to_next_statement()
    else:
      # either lambda function or regular function; let's get the closure information
      func_info = self._get_value(args[0])
      if (func_info.type() != Type.FUNC):
        super().error(ErrorType.TYPE_ERROR, "Calling something not a function", self.ip)

      closure_env = func_info.value().get_env()

      this_obj = None
      if '.' in args[0]:   # we have a method call
        this_obj_name = args[0].split('.')[0]   # x.foo yields x
        this_obj = self._get_value(this_obj_name)

      self.return_stack.append(self.ip+1)
      # Create new environment, copy args into new env
      self._create_new_environment(func_info, args[1:], closure_env, this_obj)
      self.ip = self._find_first_instruction(func_info)

  # create a new environment for a function call
  def _create_new_environment(self, func_type_value, args, closure_env, this_obj):

    # now deal with all the formal parameters, which shadow any captured variables
    funcname = func_type_value.value().get_name()  # this will be lambda:line_num for lambda functions
    formal_params = self.func_manager.get_function_info(funcname).params
    if len(formal_params) != len(args):   
      super().error(ErrorType.NAME_ERROR,f"Mismatched parameter count in call to {funcname}", self.ip)

    # create a map of formal parameters that refer to the arguments passed in
    tmp_mappings = {}
    for formal, actual in zip(formal_params,args):
      formal_name = formal[0]
      formal_typename = formal[1]
      arg = self._get_value(actual)
      if arg.type() != self.compatible_types[formal_typename]:
        super().error(ErrorType.TYPE_ERROR,f"Mismatched parameter type for {formal_name} in call to {funcname}", self.ip)
      if formal_typename in self.reference_types:
        tmp_mappings[formal_name] = arg
      else:
        tmp_mappings[formal_name] = copy.copy(arg)

    # create a new environment for the target function
    self.env_manager.push() 

    # assume closure environment has been pre-flattened before its passed in
    if closure_env != None:
      self.env_manager.import_mappings(closure_env)
    
    # add "this" object, if this is a method call;
    # v3 need to document that this can be shadowed by other parameters/locals, but not the closure
    if this_obj:
      self.env_manager.create_new_symbol(InterpreterBase.THIS_DEF)
      self.env_manager.set(InterpreterBase.THIS_DEF, this_obj)

    # now that we've added captured variables to the flattened environment, process
    # formal parameters and let them shadow any captured variables
    self.env_manager.import_mappings(tmp_mappings)

      
  def _endfunc(self, return_val = None):
    if not self.return_stack:  # done with main!
      self.terminate = True
    else:
      self.env_manager.pop()  # get rid of environment for the function
      if return_val:
        self._set_result(return_val)
      else:
        # return default value for type if no return value is specified. Last param of True enables
        # creation of result variable even if none exists, or is of a different type
        return_type = self.func_manager.get_return_type_for_enclosing_function(self.ip)
        if return_type != InterpreterBase.VOID_DEF:
          self._set_result(copy.deepcopy(self.type_to_default[return_type]))
      self.ip = self.return_stack.pop()

  def _if(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid if syntax", self.ip)
    value_type = self._eval_expression(args)
    if value_type.type() != Type.BOOL:
      super().error(ErrorType.TYPE_ERROR,"Non-boolean if expression", self.ip)
    if value_type.value():
      self._advance_to_next_statement()
      self.env_manager.block_nest()  # we're in a nested block, so create new env for it
      return
    else:
      for line_num in range(self.ip+1, len(self.tokenized_program)):
        tokens = self.tokenized_program[line_num]
        if not tokens:
          continue
        if tokens[0] == InterpreterBase.ENDIF_DEF and self.indents[self.ip] == self.indents[line_num]:
          self.ip = line_num + 1
          return
        if tokens[0] == InterpreterBase.ELSE_DEF and self.indents[self.ip] == self.indents[line_num]:
          self.ip = line_num + 1
          self.env_manager.block_nest()  # we're in a nested else block, so create new env for it
          return
    super().error(ErrorType.SYNTAX_ERROR,"Missing endif", self.ip)

  def _endif(self):
    self._advance_to_next_statement()
    self.env_manager.block_unnest()

  # we would only run this if we ran the successful if block, and fell into the else at the end of the block
  # so we need to delete the old top environment
  def _else(self):
    self.env_manager.block_unnest()   # Get rid of env for block above
    for line_num in range(self.ip+1, len(self.tokenized_program)):
      tokens = self.tokenized_program[line_num]
      if not tokens:
        continue
      if tokens[0] == InterpreterBase.ENDIF_DEF and self.indents[self.ip] == self.indents[line_num]:
          self.ip = line_num + 1
          return
    super().error(ErrorType.SYNTAX_ERROR,"Missing endif", self.ip)

  def _return(self,args):
    # do we want to support returns without values?
    return_type = self.func_manager.get_return_type_for_enclosing_function(self.ip)
    default_value_type = self.type_to_default[return_type]
    if default_value_type.type() == Type.VOID:
      if args:
        super().error(ErrorType.TYPE_ERROR,"Returning value from void function", self.ip)
      self._endfunc()  # no return
      return
    if not args:
      self._endfunc()  # return default value
      return

    #otherwise evaluate the expression and return its value
    value_type = self._eval_expression(args)
    if value_type.type() != default_value_type.type():
      super().error(ErrorType.TYPE_ERROR,"Non-matching return type", self.ip)
    self._endfunc(value_type) 

  def _while(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Missing while expression", self.ip)
    value_type = self._eval_expression(args)
    if value_type.type() != Type.BOOL:
      super().error(ErrorType.TYPE_ERROR,"Non-boolean while expression", self.ip)
    if value_type.value() == False:
      self._exit_while()
      return

    # If true, we advance to the next statement
    self._advance_to_next_statement()
    # And create a new scope
    self.env_manager.block_nest()

  def _exit_while(self):
    while_indent = self.indents[self.ip]
    cur_line = self.ip + 1
    while cur_line < len(self.tokenized_program):
      if self.tokenized_program[cur_line][0] == InterpreterBase.ENDWHILE_DEF and self.indents[cur_line] == while_indent:
        self.ip = cur_line + 1
        return
      if self.tokenized_program[cur_line] and self.indents[cur_line] < self.indents[self.ip]:
        break # syntax error!
      cur_line += 1
    # didn't find endwhile
    super().error(ErrorType.SYNTAX_ERROR,"Missing endwhile", self.ip)

  def _endwhile(self, args):
    # first delete the scope
    self.env_manager.block_unnest()
    while_indent = self.indents[self.ip]
    cur_line = self.ip - 1
    while cur_line >= 0:
      if self.tokenized_program[cur_line][0] == InterpreterBase.WHILE_DEF and self.indents[cur_line] == while_indent:
        self.ip = cur_line
        return
      if self.tokenized_program[cur_line] and self.indents[cur_line] < self.indents[self.ip]:
        break # syntax error!
      cur_line -= 1
    # didn't find while
    super().error(ErrorType.SYNTAX_ERROR,"Missing while", self.ip)

  def _define_var(self, args):
    if len(args) < 2:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid var definition syntax", self.ip)
    for var_name in args[1:]:
      if var_name == InterpreterBase.THIS_DEF:
        super().error(ErrorType.NAME_ERROR,f"This cannot be defined", self.ip)
      if self.env_manager.create_new_symbol(var_name) != SymbolResult.OK:
        super().error(ErrorType.NAME_ERROR,f"Redefinition of variable {args[1]}", self.ip)
      # is the type a valid type?
      if args[0] not in self.type_to_default:
        super().error(ErrorType.TYPE_ERROR,f"Invalid type {args[0]}", self.ip)
      # Create the variable with a copy of the default value for the type
      self.env_manager.set(var_name, copy.deepcopy(self.type_to_default[args[0]]))

    self._advance_to_next_statement()

  def _lambda_def(self, args):
   # create closure and place in resultf
    cur_env = self.env_manager.get_all_active_mappings()
    closure = LambdaClosure(self.ip, cur_env)
    self._set_result(Value(Type.FUNC, closure))  
    self._exit_lambda_def()
  
  def _exit_lambda_def(self):
    lambda_indent = self.indents[self.ip]   # lambdas can be nested, need to document this in spec!
    cur_line = self.ip + 1
    while cur_line < len(self.tokenized_program):
      if self.tokenized_program[cur_line][0] == InterpreterBase.ENDLAMBDA_DEF and self.indents[cur_line] == lambda_indent:
        self.ip = cur_line + 1
        return
      if self.tokenized_program[cur_line] and self.indents[cur_line] < self.indents[self.ip]:
        break # syntax error!
      cur_line += 1
    # didn't find endwhile
    super().error(ErrorType.SYNTAX_ERROR,"Missing endlambda", self.ip)
  
  def _endlambda(self):
    self._endfunc()
 
  def _print(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid print call syntax", self.ip)
    out = [] 
    for arg in args:
      val_type = self._get_value(arg)
      out.append(str(val_type.value()))
    super().output(''.join(out))

  def _input(self, args):
    if args:
      self._print(args)
    result = super().get_input()
    self._set_result(Value(Type.STRING, result))   # return always passed back in result
  
  def _strtoint(self, args):
    if len(args) != 1:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid strtoint call syntax", self.ip)
    value_type = self._get_value(args[0])
    if value_type.type() != Type.STRING:
      super().error(ErrorType.TYPE_ERROR,"Non-string passed to strtoint", self.ip)
    self._set_result(Value(Type.INT, int(value_type.value())))   # return always passed back in result

  def _advance_to_next_statement(self):
    # for now just increment IP, but later deal with loops, returns, end of functions, etc.
    self.ip += 1

  def _setup_default_values(self):
    self.type_to_default = {}
    self.type_to_default[InterpreterBase.INT_DEF] = Value(Type.INT,0)
    self.type_to_default[InterpreterBase.STRING_DEF] = Value(Type.STRING,'')
    self.type_to_default[InterpreterBase.BOOL_DEF] = Value(Type.BOOL,False)
    self.type_to_default[InterpreterBase.VOID_DEF] = Value(Type.VOID,None)
    self.type_to_default[InterpreterBase.FUNC_DEF] = Value(Type.FUNC,None)
    self.type_to_default[InterpreterBase.OBJECT_DEF] = Value(Type.OBJECT,{})
    self.compatible_types = {} 
    self.compatible_types[InterpreterBase.INT_DEF] = Type.INT
    self.compatible_types[InterpreterBase.STRING_DEF] = Type.STRING
    self.compatible_types[InterpreterBase.BOOL_DEF] = Type.BOOL
    self.compatible_types[InterpreterBase.FUNC_DEF] = Type.FUNC
    self.compatible_types[InterpreterBase.REFINT_DEF] = Type.INT
    self.compatible_types[InterpreterBase.REFSTRING_DEF] = Type.STRING
    self.compatible_types[InterpreterBase.REFBOOL_DEF] = Type.BOOL
    self.compatible_types[InterpreterBase.OBJECT_DEF] = Type.OBJECT
    self.reference_types = {InterpreterBase.REFINT_DEF, Interpreter.REFSTRING_DEF,
                            Interpreter.REFBOOL_DEF}
    self.type_to_result = {}
    self.type_to_result[Type.INT] = 'i'
    self.type_to_result[Type.STRING] = 's'
    self.type_to_result[Type.BOOL] = 'b'
    self.type_to_result[Type.FUNC] = 'f'
    self.type_to_result[Type.OBJECT] = 'o'  # document in v3 spec

  def _setup_operations(self):
    self.binary_op_list = ['+','-','*','/','%','==','!=', '<', '<=', '>', '>=', '&', '|']
    self.binary_ops = {}
    self.binary_ops[Type.INT] = {
     '+': lambda a,b: Value(Type.INT, a.value()+b.value()),
     '-': lambda a,b: Value(Type.INT, a.value()-b.value()),
     '*': lambda a,b: Value(Type.INT, a.value()*b.value()),
     '/': lambda a,b: Value(Type.INT, a.value()//b.value()),  # // for integer ops
     '%': lambda a,b: Value(Type.INT, a.value()%b.value()),
     '==': lambda a,b: Value(Type.BOOL, a.value()==b.value()),
     '!=': lambda a,b: Value(Type.BOOL, a.value()!=b.value()),
     '>': lambda a,b: Value(Type.BOOL, a.value()>b.value()),
     '<': lambda a,b: Value(Type.BOOL, a.value()<b.value()),
     '>=': lambda a,b: Value(Type.BOOL, a.value()>=b.value()),
     '<=': lambda a,b: Value(Type.BOOL, a.value()<=b.value()),
    }
    self.binary_ops[Type.STRING] = {
     '+': lambda a,b: Value(Type.STRING, a.value()+b.value()),
     '==': lambda a,b: Value(Type.BOOL, a.value()==b.value()),
     '!=': lambda a,b: Value(Type.BOOL, a.value()!=b.value()),
     '>': lambda a,b: Value(Type.BOOL, a.value()>b.value()),
     '<': lambda a,b: Value(Type.BOOL, a.value()<b.value()),
     '>=': lambda a,b: Value(Type.BOOL, a.value()>=b.value()),
     '<=': lambda a,b: Value(Type.BOOL, a.value()<=b.value()),
    }
    self.binary_ops[Type.BOOL] = {
     '&': lambda a,b: Value(Type.BOOL, a.value() and b.value()),
     '==': lambda a,b: Value(Type.BOOL, a.value()==b.value()),
     '!=': lambda a,b: Value(Type.BOOL, a.value()!=b.value()),
     '|': lambda a,b: Value(Type.BOOL, a.value() or b.value())
    }
    self.binary_ops[Type.FUNC] = {}  # no binary ops on functions

  def _compute_indentation(self, program):
    self.indents = [len(line) - len(line.lstrip(' ')) for line in program]

  def _find_first_instruction(self, func_type_value):
    funcname = func_type_value.value().get_name()
    func_info = self.func_manager.get_function_info(funcname)
    if not func_info:
      super().error(ErrorType.NAME_ERROR,f"Unable to locate {funcname} function")

    return func_info.start_ip
  
  # varname, true/false, integer
  def _get_value(self, token):
    if not token:
      super().error(ErrorType.NAME_ERROR,f"Empty token", self.ip)
    if token[0] == '"':
      return Value(Type.STRING, token.strip('"'))
    if token.isdigit() or token[0] == '-':
      return Value(Type.INT, int(token))
    if token == InterpreterBase.TRUE_DEF or token == Interpreter.FALSE_DEF:
      return Value(Type.BOOL, token == InterpreterBase.TRUE_DEF)
    
    # local variables cannot shadow function names - v3 not required for students to worry about
    if self._is_a_function(token):
      return Value(Type.FUNC, NamedFunctionClosure(token))

    is_object = False
    if '.' in token:
      is_object = True
      obj_tokens = token.split('.')
      token = obj_tokens[0]
      member = obj_tokens[1]

    # look in environments for variable 
    value_type = self.env_manager.get(token)
    if value_type == None:   # not found
      super().error(ErrorType.NAME_ERROR,f"Unknown variable {token}", self.ip)

    # found the variable
    if not is_object:
      return value_type
    else:  
      obj_dict = value_type.value()
      if value_type.type() != Type.OBJECT:
        super().error(ErrorType.TYPE_ERROR, f"Variable {vname} is not an object", self.ip)
      if member not in obj_dict:  # look in dictionary for this object
        super().error(ErrorType.NAME_ERROR,f"Unknown member variable {member}", self.ip)
      return obj_dict[member] # returns Value object for member variable

  def _is_a_function(self, name):
    return self.func_manager.is_function(name)

  def _set_value(self, varname, to_value_type):
    value_type = self.env_manager.get(varname)
    if value_type == None:
      super().error(ErrorType.NAME_ERROR,f"Assignment of unknown variable {varname}", self.ip)
    value_type.set(to_value_type)

  def _set_result(self, value_type):
    # always stores result in the highest-level block scope for a function, so nested if/while blocks
    # don't each have their own version of result 
    # DOCUMENT THIS!
    result_var = InterpreterBase.RESULT_DEF + self.type_to_result[value_type.type()]
    self.env_manager.create_new_symbol(result_var, True)  # create in top block if it doesn't exist
    self.env_manager.set(result_var, copy.copy(value_type))

  def _eval_expression(self, tokens):
    stack = []

    for token in reversed(tokens):
      if token in self.binary_op_list:
        v1 = stack.pop()
        v2 = stack.pop()  
        if v1.type() != v2.type():
          super().error(ErrorType.TYPE_ERROR,f"Mismatching types {v1.type()} and {v2.type()}", self.ip)
        operations = self.binary_ops[v1.type()]
        if token not in operations:
          super().error(ErrorType.TYPE_ERROR,f"Operator {token} is not compatible with {v1.type()}", self.ip)
        stack.append(operations[token](v1,v2))
      elif token == '!':
        v1 = stack.pop()
        if v1.type() != Type.BOOL:
          super().error(ErrorType.TYPE_ERROR,f"Expecting boolean for ! {v1.type()}", self.ip)
        stack.append(Value(Type.BOOL, not v1.value()))
      else:
        value_type = self._get_value(token)
        stack.append(value_type)

    if len(stack) != 1:
      super().error(ErrorType.SYNTAX_ERROR,f"Invalid expression", self.ip)

    return stack[0]


