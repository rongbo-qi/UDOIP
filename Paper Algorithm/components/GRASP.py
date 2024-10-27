import copy
import math
import random
import numpy as np
import time
from dataset_config import DatasetConfig
import pickle


class GRASP:
    def __init__(self, N_c_min=[0, 2, 2], N_c_max=[20, 2, 4], alpha=0.5, maxIterations=3, RCLsize=5, poi_id_list=[],
                 route_num=2, start_end_poi=None,
                 start_day_time=None, plan_max_time=None, dataset_config=None, user_budget=1e9, not_poi_list=[],
                 pre_set_choose=None, use_cluster=True, use_comment_num=True,
                 use_min_restaurant_gap=3*3600, mode='all', current_solution=None, exp_replace=True, tightness_w = 1, 
                 no_other_poi = False, poi_set_list = None, minimax_mode = False, use_emb = True):
        '''
            参数说明：
            N_c_min: 类别的最小访问次数 按照poi_cate_dict的顺序以list的形式给出 默认为[0, 2, 2]
            N_c_max: 类别的最大访问次数 按照poi_cate_dict的顺序以list的形式给出 默认为[10, 3, 4]
            alpha: RCL*参数α 控制RCL* score的影响程度 默认为0.5
            maxIterations: 算法最大迭代次数 默认为2
            RCLsize: RCL大小 即每次迭代时的候选列表元素个数 默认为5
            poi_id_list: 必须访问的POI列表, 以list的形式给出 默认为空列表,即无必须访问的POI,此时在getPOISet函数中会根据POI热度等信息选择考虑规划的POI
            route_num: 路线数,即旅行天数 默认为3天
            start_end_poi: 单路线起始点（酒店） 默认为None则根据必须访问的POI列表的平均经纬度寻找附近的POI作为start和end
            start_day_time: 每天开始时间 默认为None则为09:00:00,以字符串的形式给出
            plan_max_time: 单条路线最大时间 默认为None则为12小时,以小时的形式给出
            dataset_config: 数据集配置 默认为None则使用默认的数据集配置
            user_budget: 用户预算 默认为1e9
            not_poi_list: 不考虑的POI列表 默认为空列表
            pre_set_choose: 预设方案选择 默认为None
            use_cluster: 是否使用聚类处理
            use_comment_num: 计算score是否考虑comment_num即评论数
            use_min_restaurant_gap: 餐饮类别的最小时间间隔约束
            mode: 模式选择 默认为'all'，即进行全部行程规划，还有'edit'即修改行程模式
            current_solution: 当mode为'edit'时，需要给出行程规划的基础方案
            exp_replace: 是否使用替换策略 默认为True
            tightness_w: 紧凑度权重 默认为1


            输出：
                self.best_solution, self.result_start_time, self.result_wait_time,
                self.result_total_score, self.result_trans_time,
                self.result_demand_satisfied
        '''

        '''
        POI数据集合初始化
        '''
        if route_num == 0:
            route_num = 1
        if dataset_config is None:
            self.dataset_config = DatasetConfig()  # 数据集配置
        else:
            self.dataset_config = dataset_config
        self.database = self.dataset_config.database  # 数据库连接
        self.poi_dict = self.dataset_config.poi_dict  # POI数据字典
        self.poi_trans_dict = self.dataset_config.poi_trans_dict  # POI交通数据字典
        self.poi_cate_dict = self.dataset_config.poi_cate_dict  # POI类别字典
        self.poi_cate_dict_name = self.dataset_config.poi_cate_dict_name  # POI类别字典(数据库中的名称)
        self.C = list(self.poi_cate_dict.values())  # 类别列表
        self.database_func = self.dataset_config.database_func  # 数据库函数

        '''
        约束条件初始化
        '''
        self.N_c_min = N_c_min  # 类别的最小访问次数 按照poi_cate_dict的顺序以list的形式给出
        self.N_c_max = N_c_max  # 类别的最大访问次数 按照poi_cate_dict的顺序以list的形式给出
        self.m = route_num  # 路线数
        self.start = start_end_poi  # 单路线起始点（酒店）
        self.end = self.start  # 单路线终点（酒店）
        self.alpha = alpha  # RCL*参数α
        self.plan_max_time = plan_max_time
        self.maxIterations = maxIterations  # 算法最大迭代次数
        self.RCLsize = RCLsize  # RCL大小
        self.poi_id_list = poi_id_list  # 必须访问的POI列表
        self.pre_set_choose = pre_set_choose  # 预设的约束方案
        self.pre_set_poi = []  # 预设的纳入规划算法的POI列表
        self.start_day_time = start_day_time

        self.use_cluster = use_cluster  # 是否进行POI聚类
        self.not_poi_list = not_poi_list  # 不考虑的POI列表
        self.user_budget = user_budget  # 用户预算
        self.use_comment_num = use_comment_num
        self.use_N_priority = use_cluster
        self.use_min_restaurant_gap = use_min_restaurant_gap

        if self.pre_set_choose is not None:
            self.pre_set_dict = self.dataset_config.pre_set_dict[self.pre_set_choose]
            # 遍历pre_set_dict，按照key值赋值给对应的变量
            for key in self.pre_set_dict:
                setattr(self, key, self.pre_set_dict[key])

        if self.plan_max_time is None:
            self.plan_max_time = 3600 * 12  # 单条路线最大时间
        else:
            self.plan_max_time = 3600 * self.plan_max_time

        if start_day_time is None:
            self.start_day_time = time.strptime('09:00:00', '%H:%M:%S')  # 每天开始时间
        else:
            self.start_day_time = time.strptime(start_day_time, '%H:%M:%S')

        self.start_day_time_seconds = self.start_day_time.tm_hour * 3600 + self.start_day_time.tm_min * 60 + self.start_day_time.tm_sec  # 每天开始时间的秒数

        '''
        算法记录与判别变量初始化
        '''
        self.raw_N_c_max = N_c_max
        self.raw_N_c_min = N_c_min
        # self.N_c_min_judge = [[0 for k in range(route_num)] for k in range(len(self.C))] # 帮助判断每条路线的类别最小约束是否满足
        # self.N_c_max_judge = [[0 for k in range(route_num)] for k in range(len(self.C))] # 帮助判断每条路线的类别最大约束是否满足
        self.m_Time_judge = [0 for k in range(route_num)]  # 帮助判断每条路线的时间约束是否满足
        for i in range(len(self.C)):
            self.N_c_min_judge = {self.C[i]: [0 for k in range(route_num)] for i in
                                  range(len(self.C))}  # 帮助判断每条路线的类别最小约束是否满足
            self.N_c_max_judge = {self.C[i]: [0 for k in range(route_num)] for i in range(len(self.C))}
            self.N_c_start_time = {self.C[i]: [[] for k in range(route_num)] for i in
                                   range(len(self.C))}  # 每条路线每个类别POI的开始时间段记录（主要控制餐厅的用餐时间段）
            self.N_c_max = {self.C[i]: N_c_max[i] for i in range(len(self.C))}  # 类别的最大访问次数
            self.N_c_min = {self.C[i]: N_c_min[i] for i in range(len(self.C))}  # 类别的最小访问次数
        self.I = []  # 必须访问的POI列表
        self.I_judge = []  # 帮助判断必须访问的POI列表是否已经访问
        self.P = []  # 所有POI列表
        if self.use_cluster:
            self.poi_C = [[] for k in range(route_num)]  # POI聚类结果
        else:
            self.poi_C = []
        self.start_poi_time = [[] for k in range(route_num)]  # 每条路线每个POI的开始时间
        self.wait_poi_time = [[] for k in range(route_num)]  # 每条路线每个POI的等待时间
        # self.N_c_start_time = [[[] for k in range(route_num)] for k in range(len(self.C))] # 每条路线每个类别POI的开始时间段记录（主要控制餐厅的用餐时间段）
        self.user_budget_judge = 0  # 用户预算约束是否满足
        self.swapped_pairs = set() # 交换对存储

        '''
        结果
        '''
        self.every_intermediate_result = []  # 每次迭代的结果
        self.best_solution = [[] for k in range(route_num)]  # 最优解
        self.result_start_time = [[] for k in range(route_num)]  # 最优解的开始时间
        self.result_wait_time = [[] for k in range(route_num)]  # 最优解的等待时间
        self.result_trans_time = [[] for k in range(route_num)]  # 最优解的转移时间
        self.result_total_score = [0 for k in range(route_num)]  # 最优解的总分
        self.result_demand_satisfied = {'N_c_min': [False for k in range(route_num)],
                                        'N_c_max': [False for k in range(route_num)],
                                        'time': [False for k in range(route_num)], 'poi_list': False}  # 最优解的约束满足情况
                
        '''
        模式
        '''
        self.mode = mode
        self.current_solution_id = current_solution
        if self.mode == 'edit':
            self.use_cluster = True
            self.maxIterations = 1
            if self.start == None:
                self.start = self.current_solution_id[0][0]

        #0706
        self.use_embb = use_emb

        '''
        0526
        '''
        # if self.use_embb:
        #     try:
        #         self.poi_emb_dict = pickle.load(open('data/embeddings_dict_cpu.pkl', 'rb'))
        #     except:
        #         self.poi_emb_dict = pickle.load(open('../data/embeddings_dict_cpu.pkl', 'rb'))
        #     self.user_emb = self.makeUserembed()
        #     self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        if self.use_embb:
            try:
                self.poi_emb = np.load('data/tianjin_poi_embeddings_20240820192032.npy')
            except:
                self.poi_emb = np.load('../data/tianjin_poi_embeddings_20240820192032.npy')
            poi_emb_poi_ids, poi_emb_embeddings = self.poi_emb[:, 0], self.poi_emb[:, 1:]
            self.poi_emb_dict = dict(zip(poi_emb_poi_ids, poi_emb_embeddings))
            self.user_emb = self.makeUserembed()

        self.exp_replace = exp_replace

        #0606
        self.current_i = 0

        #0617
        self.tightness_w = tightness_w
        self.no_other_poi = no_other_poi

        #0624
        self.poi_cluster_center_location = {}

        #0706
        self.poi_set_list = poi_set_list
        self.minimax_mode = minimax_mode

        #0707
        #固定随机种子
        random.seed(0)
        self.rest_can_visit_many_times = True

        #0711
        self.alternative_poi_list = []

        #0822
        self.catering_feature = False
        if self.N_c_min[self.poi_cate_dict['restaurant']] > 2:
            self.catering_feature = True

    def __del__(self):
        self.database.close()

    def getEditPOISet(self):
        for i in range(self.m):
            P_m = []
            if self.database_func['getOnePOI'](self.start)[self.poi_dict['poi_cate']] == self.poi_cate_dict['hotel']:
                current_poi_id_reshape = self.current_solution_id[i][1:-1]
            else:
                current_poi_id_reshape = self.current_solution_id[i]
            poi_list, _ = self.database_func['getManyPOI'](current_poi_id_reshape)
            poi_list = np.array(poi_list, dtype=object)
            poi_list = poi_list.tolist()
            P_m.extend(poi_list)
            for poi_id in current_poi_id_reshape:
                othersPOI = self.database_func['getNearestPOIs'](poi_id, 20, self.poi_cate_dict_name['attraction'])
                P_m.extend(othersPOI)
                othersPOI = self.database_func['getNearestPOIs'](poi_id, 10, self.poi_cate_dict_name['restaurant'])
                P_m.extend(othersPOI)

            poi_list, _ = self.database_func['getManyPOI'](self.poi_id_list)
            poi_list = np.array(poi_list, dtype=object)
            poi_list = poi_list.tolist()
            P_m.extend(poi_list)
            self.I.extend(poi_list)
            for poi_id in self.poi_id_list:
                othersPOI = self.database_func['getNearestPOIs'](poi_id, 20, self.poi_cate_dict_name['attraction'])
                P_m.extend(othersPOI)
                othersPOI = self.database_func['getNearestPOIs'](poi_id, 10, self.poi_cate_dict_name['restaurant'])
                P_m.extend(othersPOI)

            # 如果既没有固定酒店，第一个也不是酒店
            if self.database_func['getOnePOI'](self.start)[self.poi_dict['poi_cate']] != self.poi_cate_dict['hotel']:
                I_lon = 0
                I_lat = 0
                for poi in self.I:
                    I_lon += poi[self.poi_dict['poi_long']]
                    I_lat += poi[self.poi_dict['poi_lat']]
                I_lon = I_lon / len(self.P)
                I_lat = I_lat / len(self.P)
                self.start = \
                    self.database_func['getNearestPOIsbyPosition']([I_lon, I_lat], 1, self.poi_cate_dict_name['hotel'])[
                        0][
                        self.poi_dict['poi_id']]
                self.end = self.start
                
            othersPOI = self.database_func['getNearestPOIs'](self.start, 20, self.poi_cate_dict_name['restaurant'])
            P_m.extend(othersPOI)

            # Drop the duplicate POIs
            p_id = []
            for p in P_m:
                p_id.append(p[self.poi_dict['poi_id']])
            p_id = np.array(p_id, dtype=object)
            p_id, p_id_index = np.unique(p_id, return_index=True)
            # Exclude POIs in not_poi_list
            for poi in P_m:
                if poi[self.poi_dict['poi_id']] in self.not_poi_list:
                    p_id_index = np.delete(p_id_index, np.where(p_id == poi[self.poi_dict['poi_id']])[0])
            P_m = np.array(P_m, dtype=object)
            P_m = P_m[p_id_index]
            P_m = P_m.tolist()

            self.P.extend(P_m)
            self.poi_C[i] = P_m

        p_id = []
        for p in self.P:
            p_id.append(p[self.poi_dict['poi_id']])
        p_id = np.array(p_id, dtype=object)
        p_id = p_id.tolist()
        p_id.append(self.start)

        # Get the cluster to which each POI belongs
        poi_cluster = self.database_func['getPOICluster'](p_id)
        self.poi_cluster_dict = {}
        for poi in poi_cluster:
            self.poi_cluster_dict[poi[self.poi_dict['poi_cluster']['poi_id']]] = poi[
                self.poi_dict['poi_cluster']['poi_cluster']]
        # Get the estimated transfer time for each cluster
        poi_all_cluster = self.poi_cluster_dict.values()
        poi_all_cluster = list(set(poi_all_cluster))
        poi_trans_center = self.database_func['getPOITransCenter'](poi_all_cluster)

        #0624
        poi_centers_position = self.database_func['getPOIClusterCenterPositions'](poi_all_cluster)
        for center in poi_centers_position:
            self.poi_cluster_center_location[center[self.poi_dict['poi_cluster_position']['poi_cluster']]] = [center[self.poi_dict['poi_cluster_position']['poi_center_long']], center[self.poi_dict['poi_cluster_position']['poi_center_lat']]]

        self.poi_trans_center_dict = {}
        for tcenter in poi_trans_center:
            bestTime = min(tcenter[self.poi_dict['poi_trans_center']['poi_center_transit_time']],
                           tcenter[self.poi_dict['poi_trans_center']['poi_center_walking_time']])
            self.poi_trans_center_dict[(tcenter[self.poi_dict['poi_trans_center']['poi_from_center']],
                                        tcenter[self.poi_dict['poi_trans_center']['poi_to_center']])] = bestTime
            
        self.I_judge = [False for i in range(len(self.poi_id_list))]


    def getEditSolution(self):
        # timeWindowUsed_record = copy.deepcopy(self.N_c_start_time)
        if self.best_solution[0] == []:
            self.getEditPOISet()
            for k in range(self.m):
                self.best_solution[k].append(list(self.database.getOnePOI(self.start)))
                self.best_solution[k].append(list(self.database.getOnePOI(self.start)))
                self.N_c_max_judge[self.poi_cate_dict['hotel']][k] += 2
                self.N_c_min_judge[self.poi_cate_dict['hotel']][k] += 2
        for i in range(self.m):
            for poi_id in self.current_solution_id[i]:
                if self.current_solution_id[i].index(poi_id) == 0 or self.current_solution_id[i].index(poi_id) == len(self.current_solution_id[i]) - 1:
                    continue
                # 使用updateRoute函数更新每条路线的POI列表
                # [k, partialSolution[k][i], p, i, j, timeWindowUsed]
                p = None
                for ps in self.P:
                    if ps[self.poi_dict['poi_id']] == poi_id:
                        p = ps
                        break
                # if p == None:
                #     p = list(self.database.getOnePOI(poi_id))
                # 插入位置 前i后j
                # 确定timeWindowUsed
                timeWindowUsed = 0
                if p[self.poi_dict['poi_cate']] == self.poi_cate_dict['restaurant']:
                    timeWindowUsed = len(self.N_c_start_time[p[self.poi_dict['poi_cate']]][i]) if len(
                        self.N_c_start_time[p[self.poi_dict['poi_cate']]][i]) < len(
                        p[self.poi_dict['poi_open_time_list']]) else len(p[self.poi_dict['poi_open_time_list']])

                self.updateRoutek(self.best_solution, [i, None, p, self.current_solution_id[i].index(poi_id) - 1,
                                                       self.current_solution_id[i].index(poi_id), timeWindowUsed])

                while p in self.P:
                    self.P.remove(p)
                if self.use_cluster:
                    for poick in self.poi_C:
                        poick_id = [poi[self.poi_dict['poi_id']] for poi in poick]
                        while p[self.poi_dict['poi_id']] in poick_id:
                            poick.pop(poick_id.index(p[self.poi_dict['poi_id']]))
                            poick_id = [poi[self.poi_dict['poi_id']] for poi in poick]

                self.checkPossible2VisitNew(self.best_solution)
                self.checkSatisfiedNconstrains(self.best_solution)


    def reset(self):
        """
        Resets the state of the GRASP algorithm.

        This method resets the variables and data structures used by the GRASP algorithm to their initial state.

        Args:
            None

        Returns:
            None
        """
        route_num = self.m
        N_c_min = self.raw_N_c_min
        N_c_max = self.raw_N_c_max
        self.m_Time_judge = [0 for k in
                             range(route_num)]  # Helps determine if the time constraint for each route is satisfied
        for i in range(len(self.C)):
            self.N_c_min_judge = {self.C[i]: [0 for k in range(route_num)] for i in
                                  range(
                                      len(self.C))}  # Helps determine if the minimum category constraint for each route is satisfied
            self.N_c_max_judge = {self.C[i]: [0 for k in range(route_num)] for i in range(
                len(self.C))}  # Helps determine if the maximum category constraint for each route is satisfied
            self.N_c_start_time = {self.C[i]: [[] for k in range(route_num)] for i in
                                   range(
                                       len(self.C))}  # Records the start time intervals for each category of POI in each route (mainly controls the dining time period for restaurants)
            self.N_c_max = {self.C[i]: N_c_max[i] for i in
                            range(len(self.C))}  # Maximum number of visits for each category
            self.N_c_min = {self.C[i]: N_c_min[i] for i in
                            range(len(self.C))}  # Minimum number of visits for each category
        self.I = []  # List of POIs that must be visited
        self.I_judge = []  # Helps determine if the POIs that must be visited have been visited
        self.P = []  # List of all POIs
        self.swapped_pairs = set() # 交换对存储
        if self.use_cluster:
            self.poi_C = [[] for k in range(route_num)]  # POI clustering results
        else:
            self.poi_C = []
        self.start_poi_time = [[] for k in range(route_num)]  # Start time for each POI in each route
        self.wait_poi_time = [[] for k in range(route_num)]  # Waiting time for each POI in each route
        self.user_budget_judge = 0  # Indicates whether the user's budget constraint is satisfied

        self.best_solution = [[] for k in range(route_num)]  # Best solution
        self.result_start_time = [[] for k in range(route_num)]  # Start time for the best solution
        self.result_wait_time = [[] for k in range(route_num)]  # Waiting time for the best solution
        self.result_trans_time = [[] for k in range(route_num)]  # Transfer time for the best solution
        self.result_total_score = [0 for k in range(route_num)]  # Total score for the best solution
        self.result_demand_satisfied = {'N_c_min': [False for k in range(route_num)],
                                        'N_c_max': [False for k in range(route_num)],
                                        'time': [False for k in range(route_num)],
                                        'poi_list': False}  # Indicates whether the constraints of the best solution are satisfied

    #0706
    def getPOISetFixation(self, poi_id_list):
        poi_list, _ = self.database_func['getManyPOI'](poi_id_list)
        self.poi_list = poi_list
        self.poi_list = np.array(self.poi_list, dtype=object)
        self.poi_list = self.poi_list.tolist()
        self.I.extend(poi_list)
        self.P.extend(poi_list)

        # self.poi_set_list_set = []

        othersPOI, _ = self.database_func['getManyPOI'](self.poi_set_list)
        self.poi_set_list_set = othersPOI
        self.poi_set_list_set = self.poi_set_list_set.tolist()
        # self.P.extend(othersPOI)

        if self.start == None:
            # Calculate the average longitude and latitude of I, then find nearby POIs as start and end
            I_lon = 0
            I_lat = 0
            for poi in self.I:
                I_lon += poi[self.poi_dict['poi_long']]
                I_lat += poi[self.poi_dict['poi_lat']]
            I_lon = I_lon / len(self.I)
            I_lat = I_lat / len(self.I)
            self.start = \
                self.database_func['getNearestPOIsbyPosition']([I_lon, I_lat], 1, self.poi_cate_dict_name['hotel'])[
                    0][
                    self.poi_dict['poi_id']]
            self.end = self.start

        self.I_judge = [False for i in range(len(self.I))]

        # Drop the duplicate POIs
        p_id = []
        for p in self.P:
            p_id.append(p[self.poi_dict['poi_id']])
        p_id = np.array(p_id, dtype=object)
        p_id, p_id_index = np.unique(p_id, return_index=True)
        # Exclude POIs in not_poi_list
        for poi in self.P:
            if poi[self.poi_dict['poi_id']] in self.not_poi_list:
                p_id_index = np.delete(p_id_index, np.where(p_id == poi[self.poi_dict['poi_id']])[0])
        self.P = np.array(self.P, dtype=object)
        self.P = self.P[p_id_index]
        self.P = self.P.tolist()

        p_id = p_id.tolist()
        p_id.append(self.start)

        for p in self.poi_set_list_set:
            p_id.append(p[self.poi_dict['poi_id']])

        # if self.use_cluster:
        #     for poi in self.P_hotel_rest:
        #         p_id.append(poi[self.poi_dict['poi_id']])

        # Get the cluster to which each POI belongs
        poi_cluster = self.database_func['getPOICluster'](p_id)
        self.poi_cluster_dict = {}
        for poi in poi_cluster:
            self.poi_cluster_dict[poi[self.poi_dict['poi_cluster']['poi_id']]] = poi[
                self.poi_dict['poi_cluster']['poi_cluster']]
        # Get the estimated transfer time for each cluster
        poi_all_cluster = self.poi_cluster_dict.values()
        poi_all_cluster = list(set(poi_all_cluster))
        poi_trans_center = self.database_func['getPOITransCenter'](poi_all_cluster)

        poi_centers_position = self.database_func['getPOIClusterCenterPositions'](poi_all_cluster)
        for center in poi_centers_position:
            self.poi_cluster_center_location[center[self.poi_dict['poi_cluster_position']['poi_cluster']]] = [center[self.poi_dict['poi_cluster_position']['poi_center_long']], center[self.poi_dict['poi_cluster_position']['poi_center_lat']]]

        self.poi_trans_center_dict = {}
        for tcenter in poi_trans_center:
            if not self.minimax_mode:
                bestTime = min(tcenter[self.poi_dict['poi_trans_center']['poi_center_transit_time']],
                                tcenter[self.poi_dict['poi_trans_center']['poi_center_walking_time']])
            else:
                bestTime = min(tcenter[self.poi_dict['poi_trans_center']['poi_center_driving_time']],
                                tcenter[self.poi_dict['poi_trans_center']['poi_center_walking_time']])
            self.poi_trans_center_dict[(tcenter[self.poi_dict['poi_trans_center']['poi_from_center']],
                                        tcenter[self.poi_dict['poi_trans_center']['poi_to_center']])] = bestTime
            
        return self.P

    # 获取规划算法考虑的POI集合
    def getPOISet(self, poi_id_list):
        """
        Retrieves a set of Points of Interest (POIs) based on the given list of POI IDs.

        Args:
            poi_id_list (list): A list of POI IDs.

        Returns:
            list: A list of POIs.

        Raises:
            None

        """
        if len(poi_id_list) == 0:
            self.P.extend(self.database_func['getHotPOIs'](20, self.poi_cate_dict_name['attraction']))
            poi_id_list = [poi[self.poi_dict['poi_id']] for poi in self.P]

            poi_list, _ = self.database_func['getManyPOI'](poi_id_list)
            self.poi_list = poi_list
            self.poi_list = np.array(self.poi_list, dtype=object)
            self.poi_list = self.poi_list.tolist()

            if self.start == None:
                # Calculate the average longitude and latitude of I, then find nearby POIs as start and end
                I_lon = 0
                I_lat = 0
                for poi in self.P:
                    I_lon += poi[self.poi_dict['poi_long']]
                    I_lat += poi[self.poi_dict['poi_lat']]
                I_lon = I_lon / len(self.P)
                I_lat = I_lat / len(self.P)
                self.start = \
                    self.database_func['getNearestPOIsbyPosition']([I_lon, I_lat], 1, self.poi_cate_dict_name['hotel'])[
                        0][
                        self.poi_dict['poi_id']]
                self.end = self.start

            for poi_id in poi_id_list:
                othersPOI = self.database_func['getNearestPOIs'](poi_id, 10, self.poi_cate_dict_name['restaurant'])
                self.P.extend(othersPOI)

        else:
            poi_list, _ = self.database_func['getManyPOI'](poi_id_list)
            self.poi_list = poi_list
            self.poi_list = np.array(self.poi_list, dtype=object)
            self.poi_list = self.poi_list.tolist()
            self.I.extend(poi_list)
            id_to_object = {item[self.poi_dict['poi_id']]: item for item in self.I}
            # 使用列表推导式按照 poi_id_list 的顺序重建 self.I
            self.I = [id_to_object[poi_id] for poi_id in poi_id_list if poi_id in id_to_object]
            self.P.extend(poi_list)
            if self.pre_set_poi != None:
                pre_set_poi_list, _ = self.database_func['getManyPOI'](self.pre_set_poi)
                self.P.extend(pre_set_poi_list)

            if self.no_other_poi == False:
                for poi_id in poi_id_list:
                    othersPOI = self.database_func['getNearestPOIs'](poi_id, 20, self.poi_cate_dict_name['attraction'])
                    self.P.extend(othersPOI)
                    othersPOI = self.database_func['getNearestPOIs'](poi_id, 10, self.poi_cate_dict_name['restaurant'])
                    self.P.extend(othersPOI)

            if self.start == None:
                # Calculate the average longitude and latitude of I, then find nearby POIs as start and end
                I_lon = 0
                I_lat = 0
                for poi in self.I:
                    I_lon += poi[self.poi_dict['poi_long']]
                    I_lat += poi[self.poi_dict['poi_lat']]
                I_lon = I_lon / len(self.I)
                I_lat = I_lat / len(self.I)
                self.start = \
                    self.database_func['getNearestPOIsbyPosition']([I_lon, I_lat], 1, self.poi_cate_dict_name['hotel'])[
                        0][
                        self.poi_dict['poi_id']]
                self.end = self.start
        if self.no_other_poi == False:
            othersPOI = self.database_func['getNearestPOIs'](self.start, 20, self.poi_cate_dict_name['restaurant'])

            if not self.use_cluster:
                self.P.extend(othersPOI)
            else:
                # Store them in P_hotel_rest first, and then evenly distribute them during clustering
                self.P_hotel_rest = []
                self.P_hotel_rest.extend(othersPOI)

        self.I_judge = [False for i in range(len(self.I))]

        # Drop the duplicate POIs
        p_id = []
        for p in self.P:
            p_id.append(p[self.poi_dict['poi_id']])
        p_id = np.array(p_id, dtype=object)
        p_id, p_id_index = np.unique(p_id, return_index=True)
        # Exclude POIs in not_poi_list
        for poi in self.P:
            if poi[self.poi_dict['poi_id']] in self.not_poi_list:
                p_id_index = np.delete(p_id_index, np.where(p_id == poi[self.poi_dict['poi_id']])[0])
        self.P = np.array(self.P, dtype=object)
        self.P = self.P[p_id_index]
        self.P = self.P.tolist()

        if self.use_cluster:
            self.P_hotel_rest = np.array(self.P_hotel_rest, dtype=object)
            self.P_hotel_rest = self.P_hotel_rest.tolist()

        p_id = p_id.tolist()
        p_id.append(self.start)

        if self.use_cluster:
            for poi in self.P_hotel_rest:
                p_id.append(poi[self.poi_dict['poi_id']])

        # Get the cluster to which each POI belongs
        poi_cluster = self.database_func['getPOICluster'](p_id)
        self.poi_cluster_dict = {}
        for poi in poi_cluster:
            self.poi_cluster_dict[poi[self.poi_dict['poi_cluster']['poi_id']]] = poi[
                self.poi_dict['poi_cluster']['poi_cluster']]
        # Get the estimated transfer time for each cluster
        poi_all_cluster = self.poi_cluster_dict.values()
        poi_all_cluster = list(set(poi_all_cluster))
        poi_trans_center = self.database_func['getPOITransCenter'](poi_all_cluster)

        #0624
        poi_centers_position = self.database_func['getPOIClusterCenterPositions'](poi_all_cluster)
        for center in poi_centers_position:
            self.poi_cluster_center_location[center[self.poi_dict['poi_cluster_position']['poi_cluster']]] = [center[self.poi_dict['poi_cluster_position']['poi_center_long']], center[self.poi_dict['poi_cluster_position']['poi_center_lat']]]

        self.poi_trans_center_dict = {}
        for tcenter in poi_trans_center:
            if not self.minimax_mode:
                bestTime = min(tcenter[self.poi_dict['poi_trans_center']['poi_center_driving_time']],
                                tcenter[self.poi_dict['poi_trans_center']['poi_center_transit_time']])
            else:
                bestTime = min(tcenter[self.poi_dict['poi_trans_center']['poi_center_driving_time']],
                                tcenter[self.poi_dict['poi_trans_center']['poi_center_walking_time']])
            self.poi_trans_center_dict[(tcenter[self.poi_dict['poi_trans_center']['poi_from_center']],
                                        tcenter[self.poi_dict['poi_trans_center']['poi_to_center']])] = bestTime

        #save self
        #create file named 'self_{{time.time()}}.pkl'
        # save_dict = {
        #     'P': self.P,
        #     'poi_id_list': poi_id_list,
        #     'hotel': self.start,
        #     'poi_trans_center_dict': self.poi_trans_center_dict
        # }

        # if self.current_i == 0:
        #     pickle.dump(save_dict, open(f'poi_data/self_{time.time()}.pkl', 'wb'))

        return self.P

    def POI_kmeans(self):
        """
        Perform K-means clustering on the points of interest (POI).

        Returns:
            poi_C (list): A list of clusters, where each cluster contains the POIs assigned to it.

        Raises:
            ImportError: If the Kmeans module is not found.

        """
        from Kmeans import KMeans, KMeansNew
        X = []
        for poi in self.P:
            X.append([poi[self.poi_dict['poi_long']], poi[self.poi_dict['poi_lat']]])
        X = np.array(X)
        if len(X) < self.m:
            kmeans = KMeans(n_clusters=self.m)
        else:
            kmeans = KMeans(n_clusters=self.m)
        kmeans.fit(X)
        
        # if not self.minimax_mode:
        #     for poi in self.P:
        #         X.append([poi[self.poi_dict['poi_long']], poi[self.poi_dict['poi_lat']]])
        #     X = np.array(X)
        #     kmeans = KMeans(n_clusters=self.m)
        #     kmeans.fit(X)
        # else:
        #     for poi in self.P:
        #         X.append([poi[self.poi_dict['poi_long']], poi[self.poi_dict['poi_lat']], poi[self.poi_dict['poi_rec_time']]])
        #     X = np.array(X)
        #     kmeans = KMeansNew(n_clusters=self.m)
        #     kmeans.fit(X)
        labels = kmeans.predict(X)
        for i in range(self.m):
            self.poi_C[i] = []
        for i in range(len(labels)):
            self.poi_C[labels[i]].append(self.P[i])
        # 将self.P_hotel_rest m等分存入对应的self.poi_C
        for i in range(self.m):
            if not self.minimax_mode:
                self.poi_C[i].extend(self.P_hotel_rest[i::self.m])
            else:
                self.poi_C[i].extend(self.poi_set_list_set)
            # self.poi_C[i].extend(self.poi_list)
        # 将poi_C保存入kmeans.log文件
        # with open('kmeans.log', 'w') as f:
        #     for i in range(self.m):
        #         f.write(f'第{i}类：\n')
        #         for poi in self.poi_C[i]:
        #             f.write(f'{poi[self.poi_dict["poi_id"]]} ')
        #             f.write(f'{poi[self.poi_dict["poi_name"]]} ')
        #         f.write('\n')
        return self.poi_C
    
    # 高效经纬度距离计算
    def getDistance(self, poi1_location, poi2_location):
        """
        Calculate the distance between two points of interest (POIs) based on their longitude and latitude.
        """
        lon1, lat1 = poi1_location[0], poi1_location[1]
        lon2, lat2 = poi2_location[0], poi2_location[1]
        return math.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)
    
    # 球面距离计算
    def getDistance2(self, poi1_location, poi2_location):
        """
        Calculate the distance between two points of interest (POIs) based on their longitude and latitude.
        """
        lon1, lat1 = poi1_location[0], poi1_location[1]
        lon2, lat2 = poi2_location[0], poi2_location[1]
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371
        return c * r
    
    # def getPOIsTransTime(self, poi_from, poi_to):
    #     """
    #     Calculates the transportation time between two points of interest (POIs).

    #     Args:
    #         poi_from (dict): A dictionary representing the starting POI.
    #         poi_to (dict): A dictionary representing the destination POI.

    #     Returns:
    #         int: The transportation time between the two POIs.
    #     """
    #     from_cluster_id = self.poi_cluster_dict[poi_from[self.poi_dict['poi_id']]]
    #     to_cluster_id = self.poi_cluster_dict[poi_to[self.poi_dict['poi_id']]]

    #     res_middle = self.poi_trans_center_dict[(from_cluster_id, to_cluster_id)]

    #     from_location = [poi_from[self.poi_dict['poi_long']], poi_from[self.poi_dict['poi_lat']]]
    #     to_location = [poi_to[self.poi_dict['poi_long']], poi_to[self.poi_dict['poi_lat']]]

    #     from_cluster_center_location = self.poi_cluster_center_location[from_cluster_id]
    #     to_cluster_center_location = self.poi_cluster_center_location[to_cluster_id]

    #     res_from = self.getDistance2(from_location, from_cluster_center_location)
    #     res_to = self.getDistance2(to_location, to_cluster_center_location)

    #     # Calculate direction vectors
    #     def direction_vector(start, end):
    #         return [end[0] - start[0], end[1] - start[1]]

    #     vector_center_to_from = direction_vector(from_cluster_center_location, from_location)
    #     vector_center_to_to = direction_vector(to_cluster_center_location, to_location)
    #     vector_between_centers = direction_vector(from_cluster_center_location, to_cluster_center_location)

    #     # Calculate dot products to determine if vectors are aligned or opposite
    #     def dot_product(v1, v2):
    #         return v1[0] * v2[0] + v1[1] * v2[1]

    #     if dot_product(vector_center_to_from, vector_between_centers) < 0:
    #         res_from = -res_from

    #     if dot_product(vector_center_to_to, vector_between_centers) < 0:
    #         res_to = -res_to

    #     # Convert distances to time
    #     res_from = res_from * 540
    #     res_to = res_to * 540

    #     return res_middle + res_from + res_to
    

    def getPOIsTransTime(self, poi_from, poi_to):
        """
        Calculates the transportation time between two points of interest (POIs).

        Args:
            poi_from (dict): A dictionary representing the starting POI.
            poi_to (dict): A dictionary representing the destination POI.

        Returns:
            int: The transportation time between the two POIs.
        """
        from_cluster_id = self.poi_cluster_dict[poi_from[self.poi_dict['poi_id']]]
        to_cluster_id = self.poi_cluster_dict[poi_to[self.poi_dict['poi_id']]]

        res_middle = self.poi_trans_center_dict[(from_cluster_id, to_cluster_id)]

        from_location = [poi_from[self.poi_dict['poi_long']], poi_from[self.poi_dict['poi_lat']]]
        to_location = [poi_to[self.poi_dict['poi_long']], poi_to[self.poi_dict['poi_lat']]]

        from_cluster_center_location = self.poi_cluster_center_location[from_cluster_id]
        to_cluster_center_location = self.poi_cluster_center_location[to_cluster_id]

        res_from = self.getDistance2(from_location, from_cluster_center_location)
        res_to = self.getDistance2(to_location, to_cluster_center_location)

        dis_middle = self.getDistance2(from_location, to_location)

        # Calculate direction vectors
        def direction_vector(start, end):
            return [end[0] - start[0], end[1] - start[1]]

        vector_center_to_from = direction_vector(from_cluster_center_location, from_location)
        vector_center_to_to = direction_vector(to_cluster_center_location, to_location)
        vector_between_centers = direction_vector(from_cluster_center_location, to_cluster_center_location)

        # Calculate dot products to determine if vectors are aligned or opposite
        def dot_product(v1, v2):
            return v1[0] * v2[0] + v1[1] * v2[1]

        if dot_product(vector_center_to_from, vector_between_centers) > 0:
            res_from = -res_from

        if dot_product(vector_center_to_to, vector_between_centers) < 0:
            res_to = -res_to

        # Calculate the middle velocity(0809)
        if dis_middle == 0:
            velocity_middle = 1 / 540
        elif res_middle == 0:
            velocity_middle = 1 / 540
        else:
            # velocity_middle = 1 / 540
            velocity_middle = res_middle / dis_middle

        # Convert distances to time
        res_from = res_from / velocity_middle
        res_to = res_to / velocity_middle

        return res_middle + res_from + res_to

    def getPOIsVisitTime(self, poi_info):
        """
        Get the visit time for the Points of Interest (POIs).

        Args:
            poi_info (dict): A dictionary containing information about the POIs.

        Returns:
            int: The visit time for the POIs.

        """
        # print(poi_info[self.poi_dict['poi_rec_time']])
        # print(poi_info[self.poi_dict['poi_rec_time']] * self.tightness_w)
        if poi_info[self.poi_dict['poi_cate']] == self.poi_cate_dict['attraction']:
            return poi_info[self.poi_dict['poi_rec_time']] * self.tightness_w
        elif poi_info[self.poi_dict['poi_cate']] == self.poi_cate_dict['restaurant']:
            return poi_info[self.poi_dict['poi_rec_time']] * (1 if not self.catering_feature else 0.25)
        else:
            return poi_info[self.poi_dict['poi_rec_time']]


    def caculateTimeOfOnek(self, partialSolutionk, waitPOIsTimes):
        """
        Calculate the total time required for a single route.

        Args:
            partialSolutionk (list): A list of POIs representing a partial solution for the route.
            waitPOIsTimes (list): A list of waiting times for each POI.

        Returns:
            float: The total time required for the route.

        """
        # Recursive function to calculate the total time required for a single route
        if len(partialSolutionk) == 1:
            return self.getPOIsVisitTime(partialSolutionk[0])
        else:
            return (
                    self.getPOIsTransTime(partialSolutionk[0], partialSolutionk[1])
                    + waitPOIsTimes[0]
                    + self.getPOIsVisitTime(partialSolutionk[0])
                    + self.caculateTimeOfOnek(partialSolutionk[1:], waitPOIsTimes[1:])
            )

    def caculateTime(self, partialSolution):
        """
        Calculate the total time for each route in the partial solution.

        Args:
            partialSolution (list): A list of partial solutions, where each partial solution represents a route.

        Returns:
            list: A list of total times for each route in the partial solution.
        """
        total_time = []
        for partialSolutionk in partialSolution:
            total_time.append(
                self.caculateTimeOfOnek(partialSolutionk, self.wait_poi_time[partialSolution.index(partialSolutionk)]))
        return total_time

    def caculateTimewoWait(self, partialSolution):
        """
        Calculates the total time for each route in the partial solution.

        Args:
            partialSolution (list): A list of routes in the partial solution.

        Returns:
            list: A list of total times for each route in the partial solution.
        """
        total_time = []
        for partialSolutionk in partialSolution:
            total_time.append(self.caculateTimeOfOnek(partialSolutionk, [0 for i in range(len(partialSolutionk))]))
        return total_time

    def caculateTransTimeOfOnek(self, partialSolutionk):
        """
        Calculate the total transfer time for a single route recursively.

        Args:
            partialSolutionk (list): A list representing a partial solution of the route.

        Returns:
            int: The total transfer time for the given route.

        """
        if len(partialSolutionk) == 1:
            return 0
        else:
            return self.getPOIsTransTime(partialSolutionk[0], partialSolutionk[1]) + self.caculateTransTimeOfOnek(
                partialSolutionk[1:])

    def caculateTransTime(self, partialSolution):
        """
        Calculates the total transfer time for all routes in the partial solution.

        Args:
            partialSolution (list): A list of partial solutions.

        Returns:
            list: A list of total transfer times for each partial solution.
        """
        total_time = []
        for partialSolutionk in partialSolution:
            total_time.append(self.caculateTransTimeOfOnek(partialSolutionk))
        return total_time

    def getPOIOpenEndTime(self, poi_info):
        """
        Get the open and end time of a point of interest (POI).

        Args:
            poi_info (dict): A dictionary containing information about the POI.

        Returns:
            tuple: A tuple containing the open time and end time of the POI.

        """
        return poi_info[self.poi_dict['poi_open_time_list']], poi_info[self.poi_dict['poi_end_time_list']]

    def startPOITime(self, partialSolution, k):
        """
        Calculates the start time and wait time for each point of interest (POI) in a partial solution.

        Args:
            partialSolution (list): A list of POIs representing a partial solution.
            k (int): The index of the partial solution.

        Returns:
            tuple: A tuple containing the start time, wait time, and an error flag.
                - startTime (list): A list of start times for each POI in the partial solution.
                - waitTime (list): A list of wait times for each POI in the partial solution.
                - error (bool): A flag indicating if an error occurred during calculation.

        """
        startTime = []
        waitTime = []
        error = False
        pre_poi = None

        for i in range(len(partialSolution[k])):
            if i == 0:
                startTime.append(self.start_day_time_seconds)
                waitTime.append(0)
                pre_poi = partialSolution[k][i]
            else:
                if partialSolution[k][i][self.poi_dict['poi_id']] == pre_poi[self.poi_dict['poi_id']] and pre_poi[self.poi_dict['poi_cate']] == self.poi_cate_dict['restaurant']:
                    error = True
                    # break
                pre_poi = partialSolution[k][i]

                arriveTime_i = startTime[i - 1] + self.getPOIsTransTime(partialSolution[k][i - 1],
                                                                        partialSolution[k][i]) + self.getPOIsVisitTime(
                    partialSolution[k][i - 1])

                openTime_start, openTime_end = self.getPOIOpenEndTime(partialSolution[k][i])
                noWaitTime = False
                delteTime = []

                for s in range(len(openTime_start)):
                    if openTime_start[s] is None:
                        continue

                    arriveTime_i_datetime = arriveTime_i
                    openTime_start_datetime = openTime_start[s]
                    openTime_end_datetime = openTime_end[s]

                    if openTime_end_datetime - self.getPOIsVisitTime(partialSolution[k][i]) <= arriveTime_i:
                        delteTime.append(1e9 + 2)
                        error = True
                        break

                    if openTime_start_datetime - arriveTime_i_datetime < 0:
                        delteTime.append(1e9)
                    else:
                        delteTime.append(openTime_start_datetime - arriveTime_i_datetime)

                    if openTime_start_datetime <= arriveTime_i_datetime <= openTime_end_datetime - self.getPOIsVisitTime(
                            partialSolution[k][i]):
                        noWaitTime = True
                        waitTime.append(0)
                        break

                if not noWaitTime:
                    waitTime.append(min(delteTime))
                    minDelteTimeIndex = delteTime.index(min(delteTime))

                    if openTime_start[minDelteTimeIndex] > arriveTime_i:
                        arriveTime_i = openTime_start[minDelteTimeIndex]

                startTime.append(arriveTime_i)

        return startTime, waitTime, error

    def getBestFeasibleTripletbyGreedyTimeFunction(self, partialSolution, p, satisfiedNconstrains, mode='max',
                                                   k_use=None):
        """
        Finds the best feasible triplet by using a greedy time function.

        Args:
            partialSolution (list): The partial solution containing the current route for each route.
            p (list): The POI (Point of Interest) to be inserted into the route.
            satisfiedNconstrains (bool): Indicates whether the route satisfies the N constraints.
            mode (str, optional): The mode of operation. Defaults to 'max'.
            k_use (int, optional): The index of the route to be used. Defaults to None.

        Returns:
            tuple: A tuple containing the best triplet and the minimum additional time.

        """

        minAddTime = float('inf')
        bestTriplet = []
        if self.user_budget_judge + self.getPOIcost(p) > self.user_budget:
            return bestTriplet, minAddTime
        for k in range(self.m):
            if k_use is not None and k_use != k:
                continue

            # if p[self.poi_dict['poi_cate']] != self.poi_cate_dict['rest'] and satisfiedNconstrains == False and mode == 'min':
            #     continue

            # if p[self.poi_dict['poi_cate']] == self.poi_cate_dict['attraction'] and p[
            #     self.poi_dict['poi_id']] not in self.poi_id_list and satisfiedNconstrains == False and mode == 'min':
            #     continue
            if self.use_N_priority:
                if (p[self.poi_dict['poi_cate']] == self.poi_cate_dict['attraction']
                        and not satisfiedNconstrains and mode == 'min'):
                    continue
                #Attention：这里删除了一个mode == min的判断条件
                if p[self.poi_dict['poi_id']] not in self.poi_id_list \
                        and satisfiedNconstrains \
                        and not all(self.I_judge):
                    continue
            else:
                #Attention：这里删除了一个mode == min的判断条件
                if p[self.poi_dict['poi_cate']] == self.poi_cate_dict['attraction'] and p[
                    self.poi_dict[
                        'poi_id']] not in self.poi_id_list and not satisfiedNconstrains:
                    continue

            if self.N_c_max_judge[p[self.poi_dict['poi_cate']]][k] >= self.N_c_max[
                p[self.poi_dict['poi_cate']]] and mode == 'max':
                continue
            if self.N_c_min_judge[p[self.poi_dict['poi_cate']]][k] >= self.N_c_min[
                p[self.poi_dict['poi_cate']]] and mode == 'min':
                # print(self.N_c_min_judge)
                if p[self.poi_dict['poi_id']] not in self.poi_id_list:
                    continue

            if len(self.start_poi_time[k]) == 0:
                self.start_poi_time[k], self.wait_poi_time[k], _ = self.startPOITime(partialSolution, k)
            startPOITimes = self.start_poi_time[k]
            waitPOIsTimes = self.wait_poi_time[k]
            timeWindowUsed = 0
            for i in range(len(partialSolution[k])):
                if i + 1 == len(partialSolution[k]):
                    break
                j = i + 1
                i2pTime = self.getPOIsTransTime(partialSolution[k][i], p)
                arriveTime_p = startPOITimes[i] + i2pTime + self.getPOIsVisitTime(partialSolution[k][i])

                openTime_start, openTime_end = self.getPOIOpenEndTime(p)
                noWaitTime = False
                delteTime = []
                waitTime = 0
                for s in range(len(openTime_start)):
                    if self.minimax_mode and self.N_c_max[self.poi_cate_dict['restaurant']] == 1 and p[self.poi_dict['poi_cate']] == self.poi_cate_dict['restaurant'] and s == 0:
                        delteTime.append(1e9 + 1)
                        continue
                    if p[self.poi_dict['poi_cate']] == self.poi_cate_dict['restaurant'] and \
                            self.N_c_start_time[self.poi_cate_dict['restaurant']][k].count(s) >= 1 and \
                            self.N_c_min_judge[self.poi_cate_dict['restaurant']][k] < 2:
                        delteTime.append(1e9 + 1)
                        continue
                    if openTime_start[s] is None:
                        delteTime.append(1e9 + 1)
                        continue
                    arriveTime_p_datetime = arriveTime_p
                    openTime_start_datetime = openTime_start[s]
                    openTime_end_datetime = openTime_end[s]
                    # 如果该开放时间段的结束时间比到达时间还早，则跳过
                    if openTime_end_datetime - self.getPOIsVisitTime(
                            partialSolution[k][i]) <= arriveTime_p_datetime:
                        delteTime.append(1e9 + 1)
                        continue
                    singleDelteTime = openTime_start_datetime - arriveTime_p_datetime
                    if singleDelteTime < 0:
                        # 满足推荐访问时间的需求
                        if arriveTime_p_datetime <= openTime_end_datetime - self.getPOIsVisitTime(
                                partialSolution[k][i]):
                            noWaitTime = True
                            timeWindowUsed = s
                            break
                        # 不满足跳过
                        else:
                            delteTime.append(1e9 + 1)
                            continue
                    else:
                        delteTime.append(singleDelteTime)

                if not noWaitTime:
                    if min(delteTime) == 1e9 + 1:
                        continue
                    minDelteTimeIndex = delteTime.index(min(delteTime))
                    arriveTime_p = openTime_start[minDelteTimeIndex]
                    timeWindowUsed = minDelteTimeIndex
                    waitTime = min(delteTime)

                # 原始的插入后增加的时间
                addTime = i2pTime + self.getPOIsTransTime(p, partialSolution[k][j]) - self.getPOIsTransTime(
                    partialSolution[k][i], partialSolution[k][j]) + self.getPOIsVisitTime(p) + waitTime

                newArriveTime = arriveTime_p

                flag_min_gap = True
                for t in range(len(startPOITimes)):
                    if p[self.poi_dict['poi_cate']] == 1 and partialSolution[k][t][
                        self.poi_dict['poi_cate']] == 1 and abs(
                            newArriveTime - startPOITimes[t]) < (
                            self.use_min_restaurant_gap) and satisfiedNconstrains:
                        flag_min_gap = False
                        break
                if not flag_min_gap:
                    continue

                # 判断插入后是否影响其他点的时间并更新addTime
                judge_follow_time = False
                pre_restaurant_time = -1
                for follow in range(j, len(partialSolution[k])):
                    if waitPOIsTimes[follow] == 0:
                        newStartTime = startPOITimes[follow] + addTime
                    elif waitPOIsTimes[follow] >= addTime:
                        newStartTime = startPOITimes[follow]
                    else:
                        newStartTime = startPOITimes[follow] + addTime - waitPOIsTimes[follow]

                    if partialSolution[k][follow][self.poi_dict['poi_cate']] == 1:
                        if pre_restaurant_time != -1:
                            if newStartTime - pre_restaurant_time < self.use_min_restaurant_gap:
                                judge_follow_time = True
                                break
                        pre_restaurant_time = newStartTime

                    _, partialSolution_datetime = self.getPOIOpenEndTime(partialSolution[k][follow])
                    partialSolution_datetime = partialSolution_datetime[0]

                    if newStartTime > partialSolution_datetime - self.getPOIsVisitTime(partialSolution[k][follow]):
                        judge_follow_time = True
                        break
                    else:
                        # 由于waitTime的存在，所以需要更新后续POI的等待时间
                        addTime = newStartTime - startPOITimes[follow]

                # 如果影响了后续POI的访问时间则跳过
                if judge_follow_time:
                    continue

                # 如果超过了最大时间则跳过
                if self.m_Time_judge[k] + addTime > self.plan_max_time:
                    # if p[self.poi_dict['poi_id']] in self.poi_id_list:
                    #     print(p[self.poi_dict['poi_id']])
                    continue

                if not self.use_cluster:
                    # 加以路线时间平衡的惩罚参数，防止一边过长一边过短
                    averageTime = sum(self.m_Time_judge) / len(self.m_Time_judge)
                    if averageTime != 0:
                        punishment = 1 + (self.m_Time_judge[k] - averageTime) / averageTime
                        addTime = addTime * punishment

                # 加以waitTime的惩罚参数，防止等待时间过长
                if waitTime != 0:
                    punishment = 1 + waitTime / 3600
                    addTime = addTime * punishment

                if addTime < minAddTime:
                    minAddTime = addTime
                    # 元组 [路线k，上一个位置的POI，插入的POI，上一个位置的POI的索引，插入的POI的索引，选择的时间段]
                    bestTriplet = [k, partialSolution[k][i], p, i, j, timeWindowUsed]
        if self.minimax_mode:
            if bestTriplet == []:
                return bestTriplet, minAddTime
            return bestTriplet, minAddTime - self.getPOIsVisitTime(bestTriplet[2])
        else:
            return bestTriplet, minAddTime
    
    #0525
    def getBestFeasibleTripletbyGreedyTimeFunctionReplace(self, partialSolution, p, k_use=None):
        minAddTime = float('inf')
        bestTriplet = []

        for k in range(self.m):
            if k_use is not None and k_use != k:
                continue

            startPOITimes = self.start_poi_time[k]
            waitPOIsTimes = self.wait_poi_time[k]

            for(i, poi) in enumerate(partialSolution[k]):
                if poi[self.poi_dict['poi_cate']] != p[self.poi_dict['poi_cate']]:
                    continue
                if poi[self.poi_dict['poi_id']] in self.poi_id_list:
                    continue
                if poi[self.poi_dict['poi_cate']] == self.poi_cate_dict['attraction'] and self.getPOIscore(poi) > self.getPOIscore(p):
                    continue
                if self.user_budget_judge + self.getPOIcost(p) - self.getPOIcost(poi) > self.user_budget:
                    continue

                addTime = self.getPOIsTransTime(partialSolution[k][i - 1], p) + self.getPOIsTransTime(p, partialSolution[k][i + 1]) - self.getPOIsTransTime(partialSolution[k][i - 1], poi) - self.getPOIsTransTime(poi, partialSolution[k][i + 1]) + self.getPOIsVisitTime(p) - self.getPOIsVisitTime(poi)
                if addTime >= -60:
                    continue

                arrive_time_p = startPOITimes[i] + addTime - self.getPOIsVisitTime(p)

                openTime_start_p, openTime_end_p = self.getPOIOpenEndTime(p)
                openTime_start_poi, openTime_end_poi = self.getPOIOpenEndTime(poi)
                timeWindowUsed = -1
                for s in range(len(openTime_start_p)):
                    if self.minimax_mode and self.N_c_max[self.poi_cate_dict['restaurant']] == 1 and p[self.poi_dict['poi_cate']] == self.poi_cate_dict['restaurant'] and s == 0:
                        continue
                    if openTime_start_p[s] is None:
                        continue
                    if openTime_start_p[s] > startPOITimes[i + 1]:
                        break
                    if arrive_time_p + self.getPOIsVisitTime(p) > openTime_end_p[s]:
                        continue
                    
                    timeWindowUsed = s
                
                if timeWindowUsed == -1:
                    continue

                if addTime < minAddTime:
                    minAddTime = addTime
                    # 元组 [路线k，被替换的POI，插入的POI，被替换的POI的索引, Time]
                    bestTriplet = [k, poi, p, i, timeWindowUsed]

        return bestTriplet, minAddTime
    
    def sortRCL(self, CL, times, costs):
        """
        Sorts the Restricted Candidate List (RCL) based on the given times and costs.

        Parameters:
        CL (list): The Restricted Candidate List.
        times (list): The list of times.
        costs (list): The list of costs.

        Returns:
        list: The sorted Restricted Candidate List.
        """
        CL = np.array(CL, dtype=object)
        times = np.array(times)
        costs = np.array(costs)
        # #0721 minimax_mode先纳入距离中尽可能远的
        # if not self.minimax_mode:
        #     sorted_index = np.lexsort((costs, times))
        # else:
        #     sorted_index = np.lexsort((costs, -times))
        sorted_index = np.lexsort((costs, times))
        return CL[sorted_index]

    def getTopRCL(self, RCL, RCLsize):
        """
        Returns the top RCLsize elements from the given RCL (Restricted Candidate List).

        Parameters:
        RCL (list): The Restricted Candidate List.
        RCLsize (int): The number of elements to be returned.

        Returns:
        list: The top RCLsize elements from the RCL.
        """
        return RCL[0:RCLsize]

    #0525
    def getPOIembed(self, poi_info):
        if self.use_embb:
            return self.poi_emb_dict[poi_info[self.poi_dict['poi_id']]]
        else:
            return np.zeros(1024)

    #0525
    def makeUserembed(self):
        poi_prefers = []
        if len(self.poi_id_list) == 0:
            poi_prefers.extend(([0]))
        else:
            poi_prefers = self.poi_id_list
        poi_not_prefers = self.not_poi_list
        user_embed = np.zeros(self.poi_emb_dict[poi_prefers[0]].shape)
        for poi in poi_prefers:
            user_embed += self.poi_emb_dict[poi]

        for poi in poi_not_prefers:
            user_embed -= self.poi_emb_dict[poi]

        user_embed = user_embed / (len(poi_prefers) + len(poi_not_prefers))
        return user_embed

    def cosine_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def getPOIscore(self, poi_info):
        """
        Calculates the score of a point of interest (POI) based on the given POI information.

        Args:
            poi_info (dict): A dictionary containing information about the POI.

        Returns:
            float: The calculated score of the POI.
        """
        #0525
        #0706
        # if self.use_embb:
        #     cos_sim = self.cos(torch.tensor(self.user_emb), torch.tensor(self.getPOIembed(poi_info)))
        #     sim = cos_sim.item()
        # else:
        if self.use_embb:
            sim = self.cosine_similarity(self.user_emb, self.getPOIembed(poi_info))
        else:
            sim = 1
        if self.use_comment_num:
            return sim * poi_info[self.poi_dict['poi_score']] * math.log2(poi_info[self.poi_dict['poi_comment_num']] + 1)
        else:
            return sim * poi_info[self.poi_dict['poi_score']]

    def getPOIcost(self, poi_info):
        """
        Calculate the cost of a point of interest (POI).

        Parameters:
        - poi_info (dict): A dictionary containing information about the POI.

        Returns:
        - float: The cost of the POI.
        """
        return 0
        # return poi_info[self.poi_dict['poi_cost']]

    def getRCL_star(self, RCL, alpha):
        """
        Returns the Restricted Candidate List (RCL) with the highest scores based on a given alpha value.

        Parameters:
        - RCL (list): The Restricted Candidate List.
        - alpha (float): The alpha value used to determine the size of the RCL.

        Returns:
        - RCL_star (list): The RCL with the highest scores.

        """
        score = []
        for i in range(len(RCL)):
            score.append(self.getPOIscore(RCL[i][2]))

            if RCL[i][2][self.poi_dict['poi_id']] in self.poi_id_list:
                # 只保留必须访问的POI
                indexThis = i
                RCL = np.array(RCL)
                RCL = RCL[[indexThis]]
                return RCL
        score = np.array(score)
        sorted_index = np.argsort(score)
        RCL = np.array(RCL)
        RCL = RCL[sorted_index]
        top_alpha = math.ceil(alpha * self.RCLsize)
        RCL_star = RCL[0:top_alpha]
        return RCL_star

    def selectRandomly(self, RCL):
        """
        Selects a random element from the given Restricted Candidate List (RCL).

        Parameters:
        RCL (list): The Restricted Candidate List from which to select a random element.

        Returns:
        object: A randomly selected element from the RCL.
        """
        return RCL[np.random.randint(0, len(RCL))]

    def updateRoutek(self, partialSolution, randomTriplet):
        """
        Updates the route with a new POI based on the given random triplet.

        Args:
            partialSolution (list): The partial solution representing the current route.
            randomTriplet (list): The random triplet containing information about the new POI to be inserted.

        Returns:
            list: The updated partial solution with the new POI inserted.

        """

        # 更新路线类别的约束信息
        self.N_c_max_judge[randomTriplet[2][self.poi_dict['poi_cate']]][randomTriplet[0]] += 1
        self.N_c_min_judge[randomTriplet[2][self.poi_dict['poi_cate']]][randomTriplet[0]] += 1
        self.N_c_start_time[randomTriplet[2][self.poi_dict['poi_cate']]][randomTriplet[0]].append(randomTriplet[5])
        self.user_budget_judge += self.getPOIcost(randomTriplet[2])

        if randomTriplet[2][self.poi_dict['poi_id']] in self.poi_id_list:
            self.I_judge[self.poi_id_list.index(randomTriplet[2][self.poi_dict['poi_id']])] = True

        # 插入的POI选定开放时间段
        randomTriplet[2][self.poi_dict['poi_open_time_list']] = [
            randomTriplet[2][self.poi_dict['poi_open_time_list']][randomTriplet[5]]]
        randomTriplet[2][self.poi_dict['poi_end_time_list']] = [
            randomTriplet[2][self.poi_dict['poi_end_time_list']][randomTriplet[5]]]
        partialSolution[randomTriplet[0]].insert(randomTriplet[4], randomTriplet[2])

        # 更新开始和等待时间
        startPOITimes, waitPOIsTimes, _ = self.startPOITime(partialSolution, randomTriplet[0])
        self.start_poi_time[randomTriplet[0]] = startPOITimes
        self.wait_poi_time[randomTriplet[0]] = waitPOIsTimes

        return partialSolution
    
    #0525
    def updateRoutek_Replace(self, partialSolution, randomTriplet):
        """
        Updates the route with a new POI based on the given random triplet.

        Args:
            partialSolution (list): The partial solution representing the current route.
            randomTriplet (list): The random triplet containing information about the new POI to be inserted.
            [路线k 0, 被替换的POI 1, 插入的POI 2, 被替换的POI的索引 3, Time 4]
        Returns:
            list: The updated partial solution with the new POI inserted.
            
        """

        # 更新路线类别的约束信息
        # self.N_c_max_judge[randomTriplet[2][self.poi_dict['poi_cate']]][randomTriplet[0]] += 1
        # self.N_c_min_judge[randomTriplet[2][self.poi_dict['poi_cate']]][randomTriplet[0]] += 1
        # self.N_c_start_time[randomTriplet[2][self.poi_dict['poi_cate']]][randomTriplet[0]].append(randomTriplet[5])
        self.user_budget_judge = self.user_budget_judge + self.getPOIcost(randomTriplet[2]) - self.getPOIcost(randomTriplet[1])

        if randomTriplet[2][self.poi_dict['poi_id']] in self.poi_id_list:
            self.I_judge[self.poi_id_list.index(randomTriplet[2][self.poi_dict['poi_id']])] = True

        # 插入的POI选定开放时间段
        randomTriplet[2][self.poi_dict['poi_open_time_list']] = [
            randomTriplet[2][self.poi_dict['poi_open_time_list']][randomTriplet[4]]]
        randomTriplet[2][self.poi_dict['poi_end_time_list']] = [
            randomTriplet[2][self.poi_dict['poi_end_time_list']][randomTriplet[4]]]
        partialSolution[randomTriplet[0]][randomTriplet[3]] = randomTriplet[2]

        # 更新开始和等待时间
        startPOITimes, waitPOIsTimes, _ = self.startPOITime(partialSolution, randomTriplet[0])
        self.start_poi_time[randomTriplet[0]] = startPOITimes
        self.wait_poi_time[randomTriplet[0]] = waitPOIsTimes

        return partialSolution

    def checkPossible2VisitNew(self, partialSolution):
        """
        Checks if it is possible to visit a new location based on the given partial solution.

        Args:
            partialSolution: The partial solution representing the current state of the route.

        Returns:
            bool: True if it is possible to visit a new location, False otherwise.
        """
        if self.user_budget_judge >= self.user_budget:
            return False
        flag = []
        # check time constrain
        current_time = self.caculateTimewoWait(partialSolution)
        # current_time = self.caculateTime(partialSolution)
        for k in range(self.m):
            self.m_Time_judge[k] = current_time[k]
            if current_time[k] > self.plan_max_time:
                flag.append(False)
            else:
                flag.append(True)
        # judge all the flags
        if True in flag:
            return True
        else:
            return False

    def checkSatisfiedNconstrains(self, partialSolution):
        """
        Checks if the given partial solution satisfies the number of POIs constraints for each route.

        Args:
            partialSolution (list): A list of routes representing a partial solution.

        Returns:
            bool: True if the partial solution satisfies the constraints, False otherwise.
        """
        flag = []
        # check the number of POIs in every route
        for c in self.C:
            for k in range(len(partialSolution)):
                if self.N_c_min_judge[c][k] < self.N_c_min[c]:
                    flag.append(False)
                else:
                    flag.append(True)
        # judge all the flags
        if False in flag:
            return False
        else:
            return True

    def startTimeAll(self, partialSolution):
        """
        Calculates the start time and wait time for each partial solution.

        Args:
            partialSolution (list): A list of partial solutions.

        Returns:
            tuple: A tuple containing two lists - start_time and wait_time.
                - start_time (list): A list of start times for each partial solution.
                - wait_time (list): A list of wait times for each partial solution.
        """
        start_time = []
        wait_time = []
        for k in range(len(partialSolution)):
            start_time_k, wait_time_k, _ = self.startPOITime(partialSolution, k)
            start_time.append(start_time_k)
            wait_time.append(wait_time_k)
        return start_time, wait_time

    def totalScore(self, partialSolution):
        """
        Calculates the total score of a given partial solution.

        Args:
            partialSolution (list): A list of partial solutions.

        Returns:
            list: A list of scores for each partial solution.
        """
        score = []
        for k in range(len(partialSolution)):
            score_k = 0
            for poi in partialSolution[k]:
                score_k += self.getPOIscore(poi)
            score.append(score_k)
        return score

    def resultTransTime(self, partialSolution):
        """
        Calculates the transportation time between points of interest (POIs) in a partial solution.

        Args:
            partialSolution (list): A list of lists representing a partial solution. Each inner list contains the POIs in a specific route.

        Returns:
            list: A list of lists representing the transportation time between POIs in each route of the partial solution.
        """
        trans_time = []
        for k in range(len(partialSolution)):
            trans_time_k = []
            for i in range(len(partialSolution[k])):
                if i + 1 == len(partialSolution[k]):
                    break
                trans_time_k.append(self.getPOIsTransTime(partialSolution[k][i], partialSolution[k][i + 1]))
            trans_time.append(trans_time_k)
        return trans_time

    def GRASP(self):
        """
        Executes the Greedy Randomized Adaptive Search Procedure (GRASP) algorithm.

        Returns:
            tuple: A tuple containing the best solution found by the algorithm, along with other intermediate results.
        """
        # readData()
        self.database.connect()
        for i in range(self.maxIterations):
            self.current_i = i
            self.reset()
            if self.mode == 'edit':
                self.getEditSolution()
            for j in range(5):
                if j > 1 and not all(self.I_judge):
                    not_i_judge = [i for i in range(len(self.I_judge)) if not self.I_judge[i]]
                    not_i_judge_id = [self.I[i][self.poi_dict['poi_id']] for i in not_i_judge]
                    for item in not_i_judge_id:
                        for p_i in self.P:
                            if p_i[self.poi_dict['poi_id']] == item:
                                p_i[self.poi_dict['poi_rec_time']] /= (1 + 0.5 * j)
                        if self.use_cluster:   
                            for poick in self.poi_C:
                                for p_i in poick:
                                    if p_i[self.poi_dict['poi_id']] == item:
                                        p_i[self.poi_dict['poi_rec_time']] /= (1 + 0.5 * j)
                solution = self.GRASPFuzzyConstructionPhase(j)
                solution = self.localSearch(solution)
                self.best_solution = solution

            self.result_start_time, self.result_wait_time = self.startTimeAll(self.best_solution)
            self.result_total_score = self.totalScore(self.best_solution)
            self.result_trans_time = self.resultTransTime(self.best_solution)
            self.result_demand_satisfied['N_c_min'] = [self.N_c_min_judge[c][k] >= self.N_c_min[c] for c in self.C for k
                                                       in range(self.m)]
            self.result_demand_satisfied['N_c_max'] = [self.N_c_max_judge[c][k] <= self.N_c_max[c] for c in self.C for k
                                                       in range(self.m)]
            self.result_demand_satisfied['time'] = [self.m_Time_judge[k] <= self.plan_max_time for k in range(self.m)]
            self.result_demand_satisfied['poi_list'] = self.I_judge

            self.every_intermediate_result.append((self.best_solution, self.result_start_time, self.result_wait_time,
                                                   self.result_total_score, self.result_trans_time,
                                                   self.result_demand_satisfied))

        # 在every_intermediate_result中选择最优解
        best_score = 0
        best_index = 0
        for res in self.every_intermediate_result:
            solution = res[0]
            if not self.minimax_mode:
                score = sum(self.totalScore(solution))
                satistied = res[5].values()
                score_satistied = 0
                for s in satistied:
                    if s == True:
                        score_satistied += 1
                    else:
                        score_satistied += np.sum(s == True)
                score = score + score_satistied
            else:
                score, _, _ = self.evaluateSolution(solution)
                score = -score
            if score > best_score:
                best_score = score
                best_index = self.every_intermediate_result.index(res)

        self.database.close()
        return self.every_intermediate_result[best_index]

    def minClusterConstructive(self):
        first_complete = False
        RCLsize = self.RCLsize
        """
        Constructs a partial solution using the Minimum Cluster Construction algorithm.

        Returns:
            partialSolution (list): A list representing the partial solution, where each element is a route.
        """

        if self.best_solution[0] == []:
            if not self.minimax_mode:
                self.getPOISet(self.poi_id_list)
            else:
                self.getPOISetFixation(self.poi_id_list)
            if self.use_cluster:
                self.POI_kmeans()
            else:
                self.poi_C = self.P
            partialSolution = [[] for k in range(self.m)]  # m empty routes
            # add the start and end POI to every route
            for k in range(self.m):
                partialSolution[k].append(list(self.database.getOnePOI(self.start)))
                partialSolution[k].append(list(self.database.getOnePOI(self.end)))
                self.N_c_max_judge[self.poi_cate_dict['hotel']][k] += 2
                self.N_c_min_judge[self.poi_cate_dict['hotel']][k] += 2
            possible2VisitNew = self.checkPossible2VisitNew(partialSolution)
            satisfiedNconstrains = self.checkSatisfiedNconstrains(partialSolution)

        else:
            partialSolution = self.best_solution
            possible2VisitNew = self.checkPossible2VisitNew(partialSolution)
            satisfiedNconstrains = self.checkSatisfiedNconstrains(partialSolution)

        while possible2VisitNew and (len(self.P) > 0 or any([len(item) for item in self.poi_C])) and (not satisfiedNconstrains or not all(self.I_judge)):
            # start_time = time.time()  # 开始计时
            CL = []
            Times = []
            Costs = []
            if not self.use_cluster:
                for p in self.P:
                    feasibleTriplet, times = self.getBestFeasibleTripletbyGreedyTimeFunction(partialSolution, p,
                                                                                             satisfiedNconstrains,
                                                                                             'min')
                    if times == float('inf'):
                        continue
                    CL.append(feasibleTriplet)
                    Times.append(times)
                    Costs.append(self.getPOIcost(p))
            else:
                for k in range(self.m):
                    for p in self.poi_C[k]:
                        feasibleTriplet, times = self.getBestFeasibleTripletbyGreedyTimeFunction(partialSolution, p,
                                                                                                satisfiedNconstrains,
                                                                                                'min', k)
                        if times == float('inf'):
                            continue
                        CL.append(feasibleTriplet)
                        Times.append(times)
                        Costs.append(self.getPOIcost(p))

            if len(CL) == 0:
                if not all(self.I_judge) and not first_complete:
                    first_complete = True
                    un_ids = []
                    for i in range(len(self.I_judge)):
                        if self.I_judge[i] == False:
                            un_ids.append(self.poi_id_list[i])
                    # 筛选self.poi_list中未访问的POI
                    self.poi_list = [p for p in self.poi_list if p[self.poi_dict['poi_id']] in un_ids]
                    for i in range(self.m):
                        self.poi_C[i].extend(self.poi_list)
                    continue
                else:
                    break
            RCL = self.sortRCL(CL, Times, Costs)  # greedy time function
            RCL = self.getTopRCL(RCL, RCLsize)
            RCL_star = self.getRCL_star(RCL, self.alpha)

            randomTriplet = self.selectRandomly(RCL_star)
            partialSolution = self.updateRoutek(partialSolution, randomTriplet)
            # drop the POI from P
            while randomTriplet[2] in self.P:
                self.P.remove(randomTriplet[2])
            if self.use_cluster:   
                for poick in self.poi_C:
                    #0708 餐饮可重复
                    # if self.minimax_mode and randomTriplet[2][self.poi_dict['poi_cate']] == self.poi_cate_dict['restaurant'] and self.rest_can_visit_many_times:
                    #     break
                    poick_id = [poi[self.poi_dict['poi_id']] for poi in poick]
                    while randomTriplet[2][self.poi_dict['poi_id']] in poick_id:
                        poick.pop(poick_id.index(randomTriplet[2][self.poi_dict['poi_id']]))
                        poick_id = [poi[self.poi_dict['poi_id']] for poi in poick]

            possible2VisitNew = self.checkPossible2VisitNew(partialSolution)
            satisfiedNconstrains = self.checkSatisfiedNconstrains(partialSolution)

        return partialSolution

    def GRASPFuzzyConstructionPhase(self, count_times):
        """
        Executes the GRASP Fuzzy Construction Phase algorithm to generate a partial solution.

        Returns:
            partialSolution (list): The generated partial solution.
        """
        RCLsize = self.RCLsize
        partialSolution = self.minClusterConstructive()


        possible2VisitNew = self.checkPossible2VisitNew(partialSolution)
        satisfiedNconstrains = self.checkSatisfiedNconstrains(partialSolution)

        if count_times == 0:
            return partialSolution

        # poi_list_judge = all(self.I_judge)
        # if not poi_list_judge:
        #     # 缩短在POI列表中的POI的建议访问时间
        #     for i in range(len(self.I_judge)):
        #         if self.I_judge[i] == False:

        count_replace_debug = 0

        while possible2VisitNew and (len(self.P) > 0 or any([len(item) for item in self.poi_C])) :
            CL = []
            Times = []
            Costs = []
            replace_mode = False
            if not self.use_cluster:
                for p in self.P:
                    feasibleTriplet, times = self.getBestFeasibleTripletbyGreedyTimeFunction(partialSolution, p,
                                                                                             satisfiedNconstrains,
                                                                                             'max')
                    if times == float('inf'):
                        continue
                    CL.append(feasibleTriplet)
                    Times.append(times)
                    Costs.append(self.getPOIcost(p))
            else:
                for k in range(self.m):
                    for p in self.poi_C[k]:
                        feasibleTriplet, times = self.getBestFeasibleTripletbyGreedyTimeFunction(partialSolution, p,
                                                                                                 satisfiedNconstrains,
                                                                                                 'max', k)
                        if times == float('inf'):
                            continue
                        CL.append(feasibleTriplet)
                        Times.append(times)
                        Costs.append(self.getPOIcost(p))
            
            if self.exp_replace:
                if len(CL) == 0 and self.use_cluster:
                    replace_mode = True
                    for k in range(self.m):
                        for p in self.poi_C[k]:
                            feasibleTriplet, times = self.getBestFeasibleTripletbyGreedyTimeFunctionReplace(partialSolution, p, k)
                            if times == float('inf'):
                                continue
                            count_replace_debug = count_replace_debug + 1
                            if count_replace_debug > 1000:
                                break
                            CL.append(feasibleTriplet)
                            Times.append(times)
                            Costs.append(self.getPOIcost(p))

            if len(CL) == 0:
                break
            RCL = self.sortRCL(CL, Times, Costs)  # greedy time function
            RCL = self.getTopRCL(RCL, RCLsize)
            RCL_star = self.getRCL_star(RCL, self.alpha)
            randomTriplet = self.selectRandomly(RCL_star)
            if replace_mode:
                partialSolution = self.updateRoutek_Replace(partialSolution, randomTriplet)
            else:
                partialSolution = self.updateRoutek(partialSolution, randomTriplet)
            # drop the POI from P
            while randomTriplet[2] in self.P:
                self.P.remove(randomTriplet[2])

            if self.use_cluster:
                for poick in self.poi_C:
                    #0708 餐饮可重复
                    # if self.minimax_mode and randomTriplet[2][self.poi_dict['poi_cate']] == self.poi_cate_dict['restaurant'] and self.rest_can_visit_many_times:
                    #     break
                    poick_id = [poi[self.poi_dict['poi_id']] for poi in poick]
                    while randomTriplet[2][self.poi_dict['poi_id']] in poick_id:
                        poick.pop(poick_id.index(randomTriplet[2][self.poi_dict['poi_id']]))
                        poick_id = [poi[self.poi_dict['poi_id']] for poi in poick]

            #0721 #把被替换的poi加回P和 poi_C
            if replace_mode:
                self.P.append(randomTriplet[1])
                for k in range(self.m):
                    if randomTriplet[1] not in self.poi_C[k]:
                        self.poi_C[k].append(randomTriplet[1])

            possible2VisitNew = self.checkPossible2VisitNew(partialSolution)
            satisfiedNconstrains = self.checkSatisfiedNconstrains(partialSolution)

        return partialSolution

    def getBestTimeWindow(self, poi_info):
        if poi_info[self.poi_dict['poi_cate']] == 1:
            return [[11 * 3600, 14 * 3600], [17 * 3600, 22 * 3600]]
        else:
            poi_best_time = []
            for i in range(len(poi_info[self.poi_dict['poi_best_open_time_list']])):
                open_seconds = poi_info[self.poi_dict['poi_best_open_time_list']][i]
                end_seconds = poi_info[self.poi_dict['poi_best_end_time_list']][i]

                if open_seconds is None or end_seconds is None:
                    continue

                # # 随机生成最佳开始时间和结束时间
                # best_start_time = open_seconds + random.randint(0, 3600)  # 在开放时间基础上随机加上0到3600秒
                # best_end_time = best_start_time + random.randint(3600, end_seconds - open_seconds)  # 随机生成结束时间，至少1小时
                best_start_time = open_seconds
                best_end_time = end_seconds
                poi_best_time.append([best_start_time, best_end_time])
                
            return poi_best_time

    def evaluateSolution(self, solution):
        """
        Evaluates the given solution based on various criteria such as time window constraints, wait times, and balance penalty.

        Args:
            solution (list): The solution to be evaluated.

        Returns:
            tuple: A tuple containing the evaluation score, wait times, and an error flag.

        Raises:
            None

        """
        # 先判断是否满足时间窗约束等
        wait_time = [[] for k in range(self.m)]
        errors = [False for k in range(self.m)]
        addValue = 0
        for k in range(self.m):
            startTime, waitTime, error = self.startPOITime(solution, k)
            wait_time[k] = waitTime
            countInBestWindow = 0
            pre_restaurant = -1

            for t in range(len(startTime)):
                if solution[k][t][self.poi_dict['poi_cate']] == 1:
                    if pre_restaurant != -1:
                        if startTime[t] - startTime[pre_restaurant] < self.use_min_restaurant_gap:
                            error = True
                            break
                    pre_restaurant = t
                bestTw = self.getBestTimeWindow(solution[k][t])
                judgeInBestWindow = False
                for w in bestTw:
                    if w[0] <= startTime[t] <= w[1]:
                        judgeInBestWindow = True
                        break
                if judgeInBestWindow:
                    countInBestWindow += 1
            addValue += (len(solution[k]) - countInBestWindow) / len(solution[k])

            # checkbug = self.checkTwoRestaurant(solution)
            # if not checkbug and error == False:
            #     checkbug = True

            errors[k] = error

        total_trans_time = sum(self.caculateTransTime(solution))

        # 计算总等待时间
        total_wait_time = sum(sum(wait_times) for wait_times in wait_time)
        # 计算等待时间的平衡性，即归一化后等待时间标准差
        max_wait_time = max(max(wait_times) for wait_times in wait_time)
        if max_wait_time == 0:
            balance_penalty = 0
        else:
            balance_penalty = np.std(
                [wait_time[k][i] / max_wait_time for k in range(self.m) for i in range(len(wait_time[k]))])

        # 目标函数是总等待时间加上平衡性惩罚
        # if self.minimax_mode:
        #     return total_trans_time, wait_time, error
        # else:
        return (1 + balance_penalty) * (1 + addValue) * (0.25 * total_wait_time + total_trans_time), wait_time, any(errors)

    def localSearch(self, solution):
        """
        Perform local search to improve the given solution.

        Args:
            solution (list): The initial solution to be improved.

        Returns:
            list: The best solution found after local search.
        """
        best_solution = copy.deepcopy(solution)
        best_score, wt, error = self.evaluateSolution(best_solution)
        improvement = True
        while improvement:
            improvement = False
            for k in range(self.m):
                for i in range(1, len(solution[k]) - 2):
                    for j in range(i + 1, len(solution[k]) - 1):
                        new_solution = copy.deepcopy(solution)
                        new_solution[k][i:j + 1] = reversed(new_solution[k][i:j + 1])
                        score, wt, error = self.evaluateSolution(new_solution)

                        if error:
                            continue

                        if score < best_score:

                            best_score = score
                            best_solution = copy.deepcopy(new_solution)
                            self.wait_poi_time = wt
                            improvement = True

            # 0721 交换路线之间各个POI Swap between different routes
            # if self.minimax_mode:
            #     for k1 in range(self.m):
            #         for k2 in range(k1 + 1, self.m):
            #             for i in range(1, len(solution[k1]) - 1):
            #                 for j in range(1, len(solution[k2]) - 1):
            #                     new_solution = copy.deepcopy(solution)
            #                     # Swap points i of k1 and j of k2
            #                     if new_solution[k1][i][self.poi_dict['poi_cate']] != new_solution[k2][j][self.poi_dict['poi_cate']]:
            #                         continue
            #                     new_solution[k1][i], new_solution[k2][j] = new_solution[k2][j], new_solution[k1][i]
            #                     score, wt, error = self.evaluateSolution(new_solution)

            #                     if error:
            #                         continue

            #                     if score < best_score:
            #                         best_score = score
            #                         best_solution = copy.deepcopy(new_solution)
            #                         self.wait_poi_time = wt
            #                         improvement = True

            solution = best_solution
        return best_solution

    # def localSearch(self, solution):
    #     """
    #     Perform local search to improve the given solution.

    #     Args:
    #         solution (list): The initial solution to be improved.

    #     Returns:
    #         list: The best solution found after local search.
    #     """
    #     best_solution = copy.deepcopy(solution)
    #     best_score, wt, error = self.evaluateSolution(best_solution)
    #     improvement = True

    #     while improvement:
    #         improvement = False
    #         # Swap within the same route
    #         for k in range(self.m):
    #             for i in range(1, len(solution[k]) - 2):
    #                 for j in range(i + 1, len(solution[k]) - 1):
    #                     new_solution = copy.deepcopy(solution)
    #                     new_solution[k][i:j + 1] = reversed(new_solution[k][i:j + 1])
    #                     score, wt, error = self.evaluateSolution(new_solution)

    #                     if error:
    #                         continue

    #                     if score < best_score:
    #                         best_score = score
    #                         best_solution = copy.deepcopy(new_solution)
    #                         self.wait_poi_time = wt
    #                         improvement = True

    #         # 0721 交换路线之间各个POI Swap between different routes
    #         # Swap between different routes and then try reversing segments in both involved routes
    #         for k1 in range(self.m):
    #             for k2 in range(k1 + 1, self.m):
    #                 for i in range(1, len(solution[k1]) - 1):
    #                     for j in range(1, len(solution[k2]) - 1):
    #                         if (k1, i, k2, j) not in self.swapped_pairs and (k2, j, k1, i) not in self.swapped_pairs:
    #                             new_solution = copy.deepcopy(solution)
    #                             # Swap points i of k1 and j of k2
    #                             new_solution[k1][i], new_solution[k2][j] = new_solution[k2][j], new_solution[k1][i]
    #                             self.swapped_pairs.add((k1, i, k2, j))  # Record the swap
    #                             if new_solution[k1][i][self.poi_dict['poi_cate']] != new_solution[k2][j][self.poi_dict['poi_cate']]:
    #                                 continue

    #                             # Try reversing segments within both routes k1 and k2
    #                             for p in range(1, len(new_solution[k1]) - 2):
    #                                 for q in range(p + 1, len(new_solution[k1]) - 1):
    #                                     new_solution_temp = copy.deepcopy(new_solution)
    #                                     new_solution_temp[k1][p:q + 1] = reversed(new_solution_temp[k1][p:q + 1])
    #                                     score, wt, error = self.evaluateSolution(new_solution_temp)
    #                                     if not error and score < best_score:
    #                                         best_score = score
    #                                         best_solution = copy.deepcopy(new_solution_temp)
    #                                         self.wait_poi_time = wt
    #                                         improvement = True
    #                             for p in range(1, len(new_solution[k2]) - 2):
    #                                 for q in range(p + 1, len(new_solution[k2]) - 1):
    #                                     new_solution_temp = copy.deepcopy(new_solution)
    #                                     new_solution_temp[k2][p:q + 1] = reversed(new_solution_temp[k2][p:q + 1])
    #                                     score, wt, error = self.evaluateSolution(new_solution_temp)
    #                                     if not error and score < best_score:
    #                                         best_score = score
    #                                         best_solution = copy.deepcopy(new_solution_temp)
    #                                         self.wait_poi_time = wt
    #                                         improvement = True

    #         solution = best_solution
    #     return best_solution
    
    #检查是否有两个餐饮相邻
    def checkTwoRestaurant(self, solution):
        for k in range(self.m):
            for i in range(len(solution[k]) - 1):
                if solution[k][i][self.poi_dict['poi_cate']] == 1 and solution[k][i + 1][self.poi_dict['poi_cate']] == 1:
                    return False
        return True

    def printSolutionId(self, solution):
        """
        Prints the solution ID for each point of interest in the given solution.

        Parameters:
        - solution (list): A list of lists representing the solution. Each inner list contains points of interest.

        Returns:
        - None
        """
        for k in range(self.m):
            for poi in solution[k]:
                print(poi[0], end=' ')
            print('\n')


    def printSolutionName(self, solution):
        """
        Prints the solution Name for each point of interest in the given solution.

        Parameters:
        - solution (list): A list of lists representing the solution. Each inner list contains points of interest.

        Returns:
        - None
        """
        for k in range(self.m):
            for poi in solution[k]:
                print(poi[self.poi_dict['poi_name']], end=' ')
            print('\n')
