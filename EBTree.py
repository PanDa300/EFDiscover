

class EBTree:
    def __init__(self,RHS,Calc_func,phi):
        self.nodecnt=0
        self.RHS=RHS
        self.Calc_func=Calc_func
        self.node=["null"]
        self.lson=[-1]
        self.rson=[-1]
        self.root=0
        self.flag=[-1]   #flag -1表示目前整个树是空的;0表示正常节点;1表示空子树（FD）;2表示满子树（全部是nonFD）;3表示当前节点为叶子节点，没有属性
        self.phi=phi
        self.pop=[[]]

    def GetNewNode(self):
        self.nodecnt+=1
        self.node.append("null")
        self.lson.append(-1)
        self.rson.append(-1)
        self.flag.append(3)
        self.pop.append([])
        return self.nodecnt

    def insert_dfs(self,node,LHS,attr,attr_list):
        if self.flag[node]==1 or self.flag[node]==2:
            return
        if self.flag[node]==3:
            self.node[node] = attr
            self.flag[node] = 0
            self.lson[node] = self.GetNewNode()
            self.rson[node] = self.GetNewNode()
            self.pop[self.lson[node]]=self.pop[node]
            self.pop[self.rson[node]]=self.pop[node]
            nowLHS = LHS+[attr]
            pop_now,Score = self.Calc_func(nowLHS, self.RHS,self.pop[node])
            self.pop[self.rson[node]]=pop_now
            if Score <= self.phi:
                self.flag[self.rson[node]] = 1
            else:
                nowLHS = LHS+[attr] + attr_list
                pop_now,Score = self.Calc_func(nowLHS, self.RHS,self.pop[node])
                if Score > self.phi:
                    self.flag[self.rson[node]] = 2
            nowLHS = LHS+attr_list
            pop_now,Score = self.Calc_func(nowLHS, self.RHS,self.pop[node])
            if Score > self.phi:
                self.flag[self.lson[node]] = 2
            return
        if self.lson[node]!=-1:
            nxt_node=self.lson[node]
            self.insert_dfs(nxt_node,LHS,attr,attr_list)
        if self.rson[node]!=-1:
            nxt_node=self.rson[node]
            self.insert_dfs(nxt_node,LHS+[self.node[node]],attr,attr_list)



    def insert(self,attr,attr_list):
        rt=self.root
        if self.flag[rt]==-1:
            self.node[rt]=attr
            self.flag[rt]=0
            self.lson[rt]=self.GetNewNode()
            self.rson[rt]=self.GetNewNode()
            self.pop[self.lson[rt]]=self.pop[rt]
            self.pop[self.rson[rt]]=self.pop[rt]
            LHS=[attr]
            pop_now,Score=self.Calc_func(LHS,self.RHS,self.pop[rt])
            self.pop[self.rson[rt]]=pop_now
            if Score<=self.phi:
                self.flag[self.rson[rt]]=1
            else:
                LHS=[attr]+attr_list
                pop_now,Score=self.Calc_func(LHS,self.RHS,self.pop[rt])
                if Score>self.phi:
                    self.flag[self.rson[rt]]=2
            LHS=attr_list
            pop_now,Score=self.Calc_func(LHS,self.RHS,self.pop[rt])
            if Score>self.phi:
                self.flag[self.lson[rt]]=2

            return
        self.insert_dfs(rt,[],attr,attr_list)

    def search_dfs(self,node,LHS):
        if self.flag[node]==1:
            return [LHS]
        if self.flag[node]==2 or self.flag[node]==3:
            return []
        lsn=self.lson[node]
        rsn=self.rson[node]
        return self.search_dfs(lsn,LHS)+self.search_dfs(rsn,LHS+[self.node[node]])

    def GetEFD(self):
        rt=self.root
        if self.flag[rt]==-1:
            return []
        return self.search_dfs(rt,[])
