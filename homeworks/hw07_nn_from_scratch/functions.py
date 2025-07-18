import numpy as np


class Module(object):
    """
    Basically, you can think of a module as of a something (black box) 
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`: 

        output = module.forward(input)

    The module should be able to perform a backward pass: to differentiate the `forward` function. 
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule. 

        gradInput = module.backward(input, gradOutput)
    """

    def __init__(self):
        self.output = None
        self.gradInput = None
        self.training = True

    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.updateOutput(input)

    def backward(self, input, gradOutput):
        """
        Performs a backpropagation step through the module, with respect to the given input.

        This includes 
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput

    def updateOutput(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.

        Make sure to both store the data in `output` field and return it. 
        """

        # The easiest case:

        # self.output = input
        # return self.output

        pass

    def updateGradInput(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own input. 
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.

        The shape of `gradInput` is always the same as the shape of `input`.

        Make sure to both store the gradients in `gradInput` field and return it.
        """

        # The easiest case:

        # self.gradInput = gradOutput
        # return self.gradInput

        pass

    def accGradParameters(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass

    def zeroGradParameters(self):
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass

    def getParameters(self):
        """
        Returns a list with its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []

    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []

    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True

    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Module"


class Sequential(Module):
    """
         This class implements a container, which processes `input` data sequentially. 

         `input` is processed by each module (layer) in self.modules consecutively.
         The resulting array is called `output`. 
    """

    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []

    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, input):
        """
        Basic workflow of FORWARD PASS:

            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})   


        Just write a little loop. 
        """

        # Your code goes here. ################################################
        temp = None
        for n in range(len(self.modules)):
            temp = self.modules[n].forward(input)
            input = temp
        self.output = temp
        return self.output

    def backward(self, input, gradOutput):
        """
        Workflow of BACKWARD PASS:

            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)   
            gradInput = module[0].backward(input, g_1)   


        !!!

        To ech module you need to provide the input, module saw while forward pass, 
        it is used while computing gradients. 
        Make sure that the input for `i-th` layer the output of `module[i]` (just the same input as in forward pass) 
        and NOT `input` to this Sequential module. 

        !!!

        """
        y = input
        # Your code goes here. ################################################
        for n in range(len(self.modules)-1, 0, -1):
            for i in range(0, n+1):
                temp = self.modules[i].forward(y)
                y = temp
            g = self.modules[n].backward(y, gradOutput)
            gradOutput = g
        self.gradInput = self.modules[0].backward(input, gradOutput)
        return self.gradInput

    def zeroGradParameters(self):
        for module in self.modules:
            module.zeroGradParameters()

    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        return [x.getParameters() for x in self.modules]

    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.getGradParameters() for x in self.modules]

    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string

    def __getitem__(self, x):
        return self.modules.__getitem__(x)

    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()


class Linear(Module):

    """
    A module which applies a linear transformation 
    A common name is fully-connected layer, InnerProductLayer in caffe. 

    The module should work with 2D input of shape (n_samples, n_feature).
    """

    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        # This is a nice initialization
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        # Your code goes here. ################################################
        self.output = np.dot(input, self.W.T) + self.b
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput = np.dot(gradOutput, self.W)
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradW = np.dot(gradOutput.T, input)
        self.gradb = np.sum(gradOutput, axis=0)

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' % (s[1], s[0])
        return q


class SoftMax(Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def updateOutput(self, input):
        # start with normalization for numerical stability
        # Your code goes here. ################################################
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        for i in range(0, self.output.shape[0]):
            self.output[i] = np.exp(self.output[i])/sum(np.exp(self.output[i]))
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        tmp = np.sum(self.output * gradOutput, axis=1, keepdims=True)
        self.gradInput = self.output * (gradOutput - tmp)
        return self.gradInput

    def __repr__(self):
        return "SoftMax"


class LogSoftMax(Module):
    def __init__(self):
        super(LogSoftMax, self).__init__()

    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))

        # Your code goes here. ################################################
        for i in range(0, self.output.shape[0]):
            self.output[i] = np.log(
                np.exp(self.output[i])/sum(np.exp(self.output[i])))
        return self.output

    def updateGradInput(self, input, gradOutput):
        exp_input = np.exp(input)
        softmax = exp_input / np.sum(exp_input, axis=-1, keepdims=True)
        self.gradInput = gradOutput - softmax * \
            np.sum(gradOutput, axis=-1, keepdims=True)
        return self.gradInput

    def __repr__(self):
        return "LogSoftMax"


class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(self, alpha=0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None
        self.moving_variance = None

    def updateOutput(self, input):
        # Your code goes here. ################################################
        # use self.EPS please
        if self.training == True:
            self.moving_mean = self.moving_mean * self.alpha + \
                input.mean(axis=0, keepdims=True) * (1-self.alpha)
            self.moving_variance = self.moving_variance * \
                self.alpha + \
                input.var(axis=0, keepdims=True, ddof=1) * (1 - self.alpha)
            self.output = np.subtract(
                input, input.mean(axis=0, keepdims=True))/np.sqrt(input.var(axis=0, keepdims=True) + self.EPS)
        else:
            self.output = np.subtract(
                input, self.moving_mean)/np.sqrt(self.moving_variance + self.EPS)
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################

        mean = np.mean(input, axis=0)
        variance = np.var(input, axis=0, ddof=0)

        s1 = np.sum(gradOutput * (input - mean) * (-0.5)
                         * (variance + self.EPS)**(-1.5), axis=0)
        s2 = np.sum(gradOutput * (-1 / np.sqrt(variance + self.EPS)), axis=0) + \
            s1 * np.mean(-2 * (input - mean), axis=0)
        self.gradInput = (gradOutput / np.sqrt(variance + self.EPS)) + \
            (s1 * 2 * (input - mean) / input.shape[0]) + \
            (s2 / input.shape[0])

        return self.gradInput

    def __repr__(self):
        return "BatchNormalization"


class ChannelwiseScaling(Module):
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis=0)
        self.gradGamma = np.sum(gradOutput*input, axis=0)

    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)

    def getParameters(self):
        return [self.gamma, self.beta]

    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]

    def __repr__(self):
        return "ChannelwiseScaling"


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()

        self.p = p
        self.mask = None

    def updateOutput(self, input):
        # Your code goes here. ################################################
        if self.training == True:
            self.mask = (np.random.random(input.shape) > self.p) / (1 - self.p)
            self.output = input*self.mask
        else:
            self.output = input
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput = gradOutput * self.mask
        return self.gradInput

    def __repr__(self):
        return "Dropout"


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, input > 0)
        return self.gradInput

    def __repr__(self):
        return "ReLU"


class LeakyReLU(Module):
    def __init__(self, slope=0.03):
        super(LeakyReLU, self).__init__()

        self.slope = slope

    def updateOutput(self, input):
        # Your code goes here. ################################################
        self.output = np.maximum(input, 0)+self.slope*np.minimum(input, 0)
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput = np.where(
            (input < 0), self.slope * gradOutput, gradOutput)
        return self.gradInput

    def __repr__(self):
        return "LeakyReLU"


class ELU(Module):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()

        self.alpha = alpha

    def updateOutput(self, input):
        # Your code goes here. ################################################
        self.output = np.where(
            (input <= 0), self.alpha*(np.exp(input)-1), input)
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput = np.where(
            (input <= 0), self.alpha*np.exp(input)*gradOutput, gradOutput)
        return self.gradInput

    def __repr__(self):
        return "ELU"


class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()

    def updateOutput(self, input):
        self.output = np.log(1+np.exp(input))
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput = 1/(1+np.exp(-input))*gradOutput
        return self.gradInput

    def __repr__(self):
        return "SoftPlus"


class Criterion(object):
    def __init__(self):
        self.output = None
        self.gradInput = None

    def forward(self, input, target):
        return self.updateOutput(input, target)

    def backward(self, input, target):
        return self.updateGradInput(input, target)

    def updateOutput(self, input, target):
        """
        Function to override.
        """
        return self.output

    def updateGradInput(self, input, target):
        """
        Function to override.
        """
        return self.gradInput

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Criterion"


class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()

    def updateOutput(self, input, target):
        self.output = np.sum(np.power(input - target, 2)) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = (input - target) * 2 / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"


class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15

    def __init__(self):
        a = super(ClassNLLCriterionUnstable, self)
        super(ClassNLLCriterionUnstable, self).__init__()

    def updateOutput(self, input, target):

        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)

        # Your code goes here. ################################################
        self.output = - \
            np.mean(np.sum(target*np.log(input_clamp), axis=1), axis=0)
        return self.output

    def updateGradInput(self, input, target):

        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        # Your code goes here. ################################################
        self.gradInput = ((-target/input_clamp+(1-target)/(1-input_clamp))/input_clamp.shape[0])*target
        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterionUnstable"
    

class ClassNLLCriterion(Criterion):
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()
        
    def updateOutput(self, input, target): 
        # Your code goes here. ################################################
        self.output = -np.mean(np.sum(target*input, axis=1), axis=0)
        return self.output

    def updateGradInput(self, input, target):
        # Your code goes here. ################################################
        self.gradInput = -target/input.shape[0]
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterion"
    

def adam_optimizer(variables, gradients, config, state):  
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault('m', {})  # first moment vars
    state.setdefault('v', {})  # second moment vars
    state.setdefault('t', 0)   # timestamp
    state['t'] += 1
    for k in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
        assert k in config, config.keys()
    
    var_index = 0 
    lr_t = config['learning_rate'] * np.sqrt(1 - config['beta2']**state['t']) / (1 - config['beta1']**state['t'])
    for current_layer_vars, current_layer_grads in zip(variables, gradients): 
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            var_first_moment = state['m'].setdefault(var_index, np.zeros_like(current_grad))
            var_second_moment = state['v'].setdefault(var_index, np.zeros_like(current_grad))
            
            # <YOUR CODE> #######################################
            # update `current_var_first_moment`, `var_second_moment` and `current_var` values
            #np.add(... , out=var_first_moment)
            #np.add(... , out=var_second_moment)
            #current_var -= ...
                    # Update biased first moment estimate
            np.add(config['beta1'] * var_first_moment,
                (1 - config['beta1']) * current_grad,
                out=var_first_moment)

            # Update biased second raw moment estimate
            np.add(config['beta2'] * var_second_moment,
                (1 - config['beta2']) * np.square(current_grad),
                out=var_second_moment)

            # Compute bias-corrected first moment estimate
            m_hat = var_first_moment / (1 - config['beta1'] ** state['t'])

            # Compute bias-corrected second raw moment estimate
            v_hat = var_second_moment / (1 - config['beta2'] ** state['t'])

            # Update parameter
            current_var -= lr_t * m_hat / (np.sqrt(v_hat) + config['epsilon'])

            # small checks that you've updated the state; use np.add for rewriting np.arrays values
            assert var_first_moment is state['m'].get(var_index)
            assert var_second_moment is state['v'].get(var_index)
            var_index += 1



class MaxPool2d(Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.gradInput = None
                    
    def updateOutput(self, input):
        input_h, input_w = input.shape[-2:]
        self.max_indices = np.zeros((input.shape[0],input.shape[1],input_h//2,input_w//2))
        # your may remove these asserts and implement MaxPool2d with padding
        self.output = np.zeros((input.shape[0],input.shape[1],input_h//2,input_w//2))
        # YOUR CODE #############################
        for batch in range(input.shape[0]):
            for ch in range(input.shape[1]):
                new_h = 0
                for h in range(0,input_h,self.kernel_size):
                    new_w = 0
                    for w in range(0,input_w,self.kernel_size):
                        self.output[batch, ch, new_h, new_w] = np.max(input[batch, ch, h:h+self.kernel_size, w:w+self.kernel_size])
                        self.max_indices[batch, ch, new_h, new_w] = np.argmax(input[batch, ch, h:h+self.kernel_size, w:w+self.kernel_size])
                        new_w +=1
                    new_h += 1
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        input_h, input_w = input.shape[-2:]
        # your may remove these asserts and implement MaxPool2d with padding
        # YOUR CODE #############################
        self.gradInput = np.zeros((input.shape[0],input.shape[1],input_h, input_w))
        for batch in range(input.shape[0]):
            for ch in range(input.shape[1]):
                new_h = 0
                for h in range(0,input_h,self.kernel_size):
                    new_w = 0
                    for w in range(0,input_w,self.kernel_size):
                        window = np.array(input[batch, ch, h:h+self.kernel_size, w:w+self.kernel_size], copy=True)
                        window = window.reshape(-1)
                        window = np.zeros_like(window)
                        window[self.max_indices[batch, ch, new_h, new_w].astype(np.int32)] = gradOutput[batch, ch, new_h, new_w]
                        self.gradInput[batch, ch, h:h+self.kernel_size, w:w+self.kernel_size] = window.reshape(self.kernel_size, self.kernel_size)
                        new_w +=1
                    new_h += 1
        return self.gradInput
        
    def __repr__(self):
        q = 'MaxPool2d, kern %d, stride %d' %(self.kernel_size, self.kernel_size)
        return q
