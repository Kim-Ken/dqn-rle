frames = 0
isLearned = False
SESS = tf.Session()

with tf.device("/cpu:0"):
    parameter_server = ParameterServer()
    thread = []

    for i in range(N_WORKERS):
        thread_name = "local_thread" + str(i+1)
        thread.append(Worker_thread(thread_name=thread_name,thread_type="learning",parameter_server=parameter_server)

    thread.append(Worker_thread(thread_name="test_thread",thread_type="test",parameter_server=parameter_server)

COORD = tf.train.Coordinator()
SESS.run(tf.global_variables_initializer())

running_threads = []
for worker in thread:
    job = lambda: worker.run()
    t = threading.Thread(target=job)
    t.start()


class Worker_thread:

    def __init__(self,thread_name,thread_type,parameter_server):
        self.enviroment = Enviroment(thread_name,thread_type,parameter_server)
        self.thread_type = thread_type

    def run(self):
        while True:
            if not(isLearned) and self.thread_type is "learning":
                self.enviroment.run()

            if not(isLearned) and self.thread_type is 'test':
                time.sleep(1.0)

            if isLearned and self.thread_type is 'learning':
                time.sleep(3.0)

            if isLearned and self.thread_type is 'test':
                time.sleep(3.0)
                self.enviroment.run()


class Enviroment:
    total_reward_vec = np.zeros(10)
    count_trial_each_thread = 0

    def __init__(self,name,thread_type,parameter_server):
        self.name = name
        self.thread_type = thread_type
        self.env = gym.make(ENV)
        self.agent = Agent(name,parameter_server)

    def run(self):
        self.agent.brain.pull_parameter_server()
        global frames
        global isLearned

        if (self.thread_type is 'test') and(self.count_trial_each_thread==0):
            self.env.reset()

        s = self.env.reset()
        R = 0
        step = 0
        while True:
            if self.thread_type is 'test':
                self.env.render()
                time.sleep(0.1)

            a = self.agent.act(s)
            s_,r,done,info = self.env.step(a)
            step +=1
            frames +=1

            r=0
            if done:
                s_ = None
                if step < 199:
                    r = -1
                else:
                    r =1

            self.agent.advance_push_local_brains(s,a,r,s_)
            s = s_
            if done or(step%Tmax==0):
                self.agent.brain.update_parameter_server()
                self.agent.brain.pull_parameter_server()

            if done:
                self.total_reward_vec = np.hstack((self.total_reward_vec[1:],step))
                self.count_trial_each_thread += 1
                break

        print("thrad" + self.name+",kaisu:"+str(self.count_trial_each_thread)+\
        "step num" +str(step)+" averstep:"+str(self.total_reward_vec.mean()))

        if self.total_reward_vec>199:
            isLearned = True
            time.sleep(2.0)
            self.agent.brain.push_parameter_server()


class Agent:
    def __init__(self,name,parameter_server):
        self.brain = LocalBrain(name,parameter_server)
        self.memory = []
        self.R =0.


    def act(self,s):
        if frames >=EPS_STEPS:
            eps = EPS_END
        else:
            eps = EPS_START + frames * (EPS_END - EPS_START)/EPS_STEPS

        if random.random() <eps:
            return random.randint(0,NUM_ACTIONS - 1)
        else:
            s = np.arrays([s])
            p = self.brain.predic_p(s)

            a = np.random.choice(NUM_ACTIONS,p=p[0])

            return a

    def advantage_push_local_brain(self,s,a,r,s_):
        def get_sample(memory,n):
            s,a,_,_ = memory[0]
            _,_,_,s_ = memory[n-1]
            return s,a,self.R,s_

        a_cats np.zeros(NUM_ACTIONS)
        a_cats[a] =1
        self.memory.append((s,a_cats,r,s_))

        self.R = (self.R + r*GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s,a,r,s_ = get_sample(self.memory,n)
                self.brain.train_push(s,a,r,s_)
                self.R = (self.R - self.memory[0][2])/GAMMA
                self.memory.pop(0)
            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s,a,r,s_ = get_sample(self.memory,N_STEP_RETURN)
            self.brain.train_push(s,a,r,s_)
            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)


class ParameterServer:
    def __init__(self):
        with ft.variable_scope("parameter_server"):
            self.model = self._build_model()

        self.get_weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="parameter_server")
        self.optimizer = tf.train.RMSPropOptimizer(LEARING_RATE,RMSPropDeclay)


    def _build_model(self):
        l_input = Input(batch_shape=(None,NUM_STATES))
        l_dense = Dense(16,activation='relu')(l_input)
        out_actions =Dense(NUM_ACTIONS,activation='softmax')
        out_value = Dense(1,activation="linear")(l_dense)
        model= Model(inputs=[l_input],outputs=[out_actions,out_value])
        plot_model=(model,to_file='A3C.png',show_shapes=True)
        return model
