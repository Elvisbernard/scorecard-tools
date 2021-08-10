# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import bisect
import numpy as np



class grouping(object):

    def __init__(self, df, var, values):
        self.dic = {}
        self.var_name = str(var)
        self.WoE = {}
        self.values = values
        self.df = df

    def addGroup(self, GO):
        if GO.label() not in self.dic:
            self.dic[GO.label()] = {}
        else:
            return 'already in it'
        self.addtogroup(GO.label(), GO.dic.copy())
    
    def valist(self):
        return list(self.dic.keys())

    def renamegroup(self, name):
        if len(self.dic[name]) == 0:
            del self.dic[name]
            return
        label = str(min(self.dic[name]))+'-->'+str(max(self.dic[name]))
        self.dic[label] = self.dic[name]
        if name!= label:
            del self.dic[name]

    def addtogroup(self, group, value):
        if group not in self.dic:
            raise ValueError('Group unkown')
        if value.__class__ == dict:
            self.dic[group].update(value)
        else:
            raise ValueError('dictionnary expected to be passed')

#   useless piece of code
#    def removefromgroup(self, group, value):
#        if group not in self.dic:
#            raise ValueError ('Group unkown')
#        if value.__class__!=list:
#            self.dic[group].pop(value)
#        else:
#            if set(value).issubset(self.dic[group]):
#                self.dic[group] = self.dic[group].difference(set(value))
#            else:
#                raise ValueError ('item not present in group')

    def movetogroup(self, src, dest, value):
        if value.__class__ == list:
            for elem in value:
                _temp = self.dic[src].pop(elem)
                self.dic[dest].update({elem:_temp})
        else:
            _temp = self.dic[src].pop(value)
            self.dic[dest].update({value:_temp})

        self.renamegroup(dest)
        self.renamegroup(src)

    def __str__(self):

        self.calc_WoE()
        #calculate the total number of good and bad in order to calculate the volume
        totgood = 0
        totbad = 0
        for k, v in self.dic.items():
            for ka, va in self.dic[k].items():
                totgood += va[0]
                totbad += va[1]

        print('{:<40s}{:>8s}{:>8s}{:>10s}{:>12s}{:>8s}'.format('Value','Good','Bad','Vol','Bad Rate', 'WoE'))


        #sort the different group in increasing order then print the different characteristics
        #for each group
        for (x, k1) in sorted([(b, a) for a, b in zip(list(self.dic.keys()), [float(i.split('--')[0]) for i in self.dic.keys()])]):
            v1 = self.dic[k1]
            
            bad = 0
            good = 0
            for k2, v2 in v1.items():
                bad += v2[1]
                good += v2[0]
            vol=str(round(100*(good+bad)/(totgood+totbad), 1))+' %'
            bady=str(int(bad))
            goody=str(int(good))
            BR= str(round(bad/(bad+good), 2))
            woe=str( round(self.WoE[k1][2],2))
            print('{:<40s}{:>8s}{:>8s}{:>10s}{:>12s}{:>8s}'.format(k1+':',goody,bady,vol,BR, woe))
        print('Information value: {}'.format(round(self.IV, 2)))
        print('Gini: {}'.format(round(self.gini, 2)))
        return str()

    def printdetail(self):
        '''
        print the different values in each group and their characteristics
        '''
        #sort the different group in increasing order then, for each group,  print the different characteristics for each observation
        for (x, k1) in sorted([(b, a) for a, b in zip(list(self.dic.keys()), [float(i.split('--')[0]) for i in self.dic.keys()])]):
            v1 = self.dic[k1]
            print('\ngroup ', k1, ':')
            print('Value\tgood\tbad\tbad rate')
            bad = 0
            good = 0


            for k2 in sorted(v1):
                v2 = self.dic[k1][k2]
                bad += v2[1]
                good += v2[0]
                print(k2, ':\t', round(v2[0]), '\t', round(v2[1]), '\t', round(v2[1]/(v2[1]+v2[0]), 2))
            print('total:\t', round(good), '\t', round(bad), '\t', round(bad/(bad+good), 2))
        return str()

    def printgroup(self, group):
        '''
        take an unique group name as argument
        print the same data as printdetail() but only for the group mentionned
        '''

        print('\ngroup ', group, ':')
        print('Value\tgood\tbad\tbad rate')
        bad = 0
        good = 0

        for k2 in sorted(self.dic[group]):
            v2 = self.dic[group][k2]
            bad += v2[1]
            good += v2[0]
            print(k2, ':\t', int(v2[0]), '\t', int(v2[1]), '\t', round(v2[1]/(v2[1]+v2[0]), 2))
        print('total:\t', int(good), '\t', int(bad), '\t', round(bad/(bad+good), 2))
        return str()

    def movegroup(self, src, dest=None, mini=None, maxi=None):
        '''
        This method takes 4 arguments:
            src :the group from where the observations come
            dest:the group where the observations should be merged in 
                    (if omitted, a new group will be created)
            mini:the lower boundary of the observations moving 
                    (optional, if omitted, the lower boundary will be used)
            maxi:the upper boundary of the observation moving 
                    (optional, if omitted, the lower boundary will be used)
        '''
        if mini == None:
            mini = min(self.dic[src].keys())

        if maxi == None:
            maxi = max(self.dic[src].keys())


        if dest == None:
            self.dic['temp'] = {}
            dest = 'temp'
        if maxi < mini:
            (maxi, mini) = (mini, maxi)
            print('max ({}) and min ({}) have been inverted'.format(mini, maxi))
        if mini == maxi:
            self.movetogroup(src, dest, mini)
        else:
            tomove = [a for a in self.dic[src].keys() if mini <= a <= maxi]
            for value in tomove:
                _temp = self.dic[src].pop(value)
                self.dic[dest].update({value:_temp})
            self.renamegroup(dest)
            self.renamegroup(src)

    def calc_WoE(self):
        '''
        WARNING: an approximation is made in the case of pure group where either the number of
        good or bad is zero by counting at least one bad or one good

        '''
        if self.defval==None:
            self.defval=[]
            
        self.tot_bad = 0
        self.tot_good = 0
        self.WoE = {}
        CumGood = [0]
        CumBad = [0]
        bad_rate = [-1]


        for k1, v1 in self.dic.items():
            self.WoE[k1] = []
            bad = 0
            good = 0
            self.IV = 0
            for k2, v2 in v1.items():
                self.tot_bad += v2[1]
                bad += v2[1]
                self.tot_good += v2[0]
                good += v2[0]
            self.WoE[k1] += [good, bad]
            CumBad += [bad]
            CumGood += [good]
            bad_rate += [bad/(bad+good)]
        for k, v in self.WoE.items():
            self.WoE[k] += [np.log((max(1, self.WoE[k][0])/self.tot_good)/(max(1, self.WoE[k][1])/self.tot_bad))]
            self.IV += ((self.WoE[k][0]/self.tot_good)-(self.WoE[k][1]/self.tot_bad))*np.log((max(1, self.WoE[k][0])/self.tot_good)/(max(1, self.WoE[k][1])/self.tot_bad))

        sortedcum = sorted(zip(bad_rate, CumGood, CumBad))
        (bad_rate, CumGood, CumBad) = list(zip(*sortedcum))
        CumGood = np.array(CumGood).cumsum()/self.tot_good
        CumBad = np.array(CumBad).cumsum()/self.tot_bad

        self.gini = 0
        for i in range(1, len(CumGood)):
            current = (CumGood[i], CumBad[i])
            previous = (CumGood[i-1], CumBad[i-1])
            self.gini += (current[1]+previous[1])*(current[0]-previous[0])
        self.gini=(1-abs(self.gini))*100

    def transform(self, path, calc=True, group=False):
        '''
        Method that matches the different values in the dataset used for the autogrouping and
        store an ordered list of weight of evidence in the dataframe mentionned by path.
        The method that take 3 arguments:
            path:    name of the dataset where the WoE should be store
            calc:    boolean indicating if the WoE should be calculated before mapping it to the data
                     calc=False allows to use WoE that has been imported
                     the default value is True
            group:   boolean indicating if a variable storing the group to which the observation belong should be created
        '''
        #calculate the Weight of Evidence if needed
        if calc == True:
            self.calc_WoE()

        #raise an error in case where no Weight of evidence has been calculated
        elif len(self.WoE) == 0:
            raise ValueError('Weight of evidence should be calculted')

        #create the name of the new variable and the list to store the values
        name = self.var_name+'_WoE'
        transformed = []
        transname = []

        #create an alphabetically ordered list of the group (needed for the bissection)
        list_SN = []
        list_key = []
        list_def ={}
        for (short_name, key) in sorted([(b, a) for a, b in zip(list(self.dic.keys()), [float(i.split('-->')[1]) for i in self.dic.keys()])]):
            if short_name not in self.defval:
                list_SN += [short_name]
                list_key += [key]
            else:
                list_def[short_name]=key

        #match the values used for the autogrouping with their Weight of Evidence
        for elem in self.values:
            if elem in self.defval:
                transformed += [self.WoE[list_def[elem]][2]]
                transname += [list_def[elem]]
            else:
                if elem in list_SN:
                    index = list_SN.index(elem)
                else:
                    index = bisect.bisect_left(list_SN, elem)
                if index == len(list_SN):
                    index += -1
                transformed += [self.WoE[list_key[index]][2]]
                transname += [list_key[index]]

        #add the list of Weight of evidence to the dataframe specified by path
        path[name] = transformed
        if group==True:
            namey=self.var_name+'_grp'
            path[namey] = transname

    def apply(self, src, dest, calc=True, group=False):
        '''
        Method that match the different values in a dataset specify by src and
        store an ordered list of weight of evidence (calculated on the autogrouping) in the dataframe mentionned by dest.
        The method that take 4 arguments:
            src:     specification of the value that should be match to the Weight of Evidence calculated
                     Format expected: DataFrame['name_of_the_variable']
            dest:    name of the dataset where the WoE should be store
            calc:    bolean indicating if the WoE should be calculated before mapping it to the data
                     calc=False allows to use WoE that has been imported
                     the default value is True
            group:   boolean indicating if a variable storing the group to which the observation belong should be created

        Note the value used to calculate the Weight of Evidence are the Values used when creating the object and not the one specified in the method
        '''

        #calculate the Weight of Evidence if needed
        if calc == True:
            self.calc_WoE()

        #raise an error in case where no Weight of evidence has been calculated
        elif len(self.WoE) == 0:
            raise ValueError('Weight of evidence should be calculted')

        #create the name of the new variable and the list to store the values
        name = self.var_name+'_WoE'
        transformed = []
        transname = []

        #create an alphabetically ordered list of the group (needed for the bissection)
        list_SN = []
        list_key = []
        list_def ={}
        for (short_name, key) in sorted([(b, a) for a, b in zip(list(self.dic.keys()), [float(i.split('-->')[1]) for i in self.dic.keys()])]):
            if short_name not in self.defval:
                list_SN += [short_name]
                list_key += [key]
            else:
                list_def[short_name]=key

        #match the values used for the autogrouping with their Weight of Evidence
        for elem in src:
            if elem in self.defval:
                transformed += [self.WoE[list_def[elem]][2]]
                transname += [list_def[elem]]
            else:
                if elem in list_SN:
                    index = list_SN.index(elem)
                else:
                    index = bisect.bisect_left(list_SN, elem)
                if index == len(list_SN):
                    index += -1
                transformed += [self.WoE[list_key[index]][2]]
                transname += [list_key[index]]

        #add the list of Weight of evidence to the dataframe specified by path
        dest[name] = transformed
        if group==True:
            namey=self.var_name+'_grp'
            dest[namey] = transname

    def graph(self, calc=True, save=False, name='group'):
        '''
        Method that creates a graph with the Weight of Evidence for the different groups
        The method takes three arguments:
            calc:    bolean indicating if the WoE should be calculated before mapping it to the data
                     calc=False allows to use WoE that has been imported
                     the default value is True
            save:    bolean indicating if the graph generated should be saved
            name:    string containing the name of the file (without extension)
        '''
        import matplotlib.pyplot as plt
        #calculate the Weight of Evidence if needed
        if calc == True:
            self.calc_WoE()

        #raise an error in case where no Weight of evidence has been calculated
        elif len(self.WoE) == 0:
            raise ValueError('Weight of evidence should be calculted')

        #create the graph
        fig=plt.figure(figsize=(6,8))
        fig.subplots_adjust(bottom=0.5)
        plt.xticks(rotation=90)
        #for each group in a list in ascending oder, plot the Weight of Evidence
        for (x, k1) in sorted([(b, a) for a, b in zip(list(self.WoE.keys()), [float(i.split('-->')[0]) for i in self.WoE.keys()])]):
            v1 = self.WoE[k1]
            plt.plot(k1, v1[2], '.', c='r')
        plt.xlabel('Groups')
        plt.ylabel('Weight of evidence')
        plt.title('Weight of evidence for the variable {}'.format(self.var_name))
        plt.show()
        if save:
            fig.savefig(name+'.png')

    def compare(self, other, save=False, name='compare'):
        '''
        Method that creates a graph with the Weight of Evidence of the different groups
        for two different groupings
        The method takes three arguments:
            other:   grouping object we want to compare to
            save:    bolean indicating if the graph generated should be saved
            name:    string containing the name of the file (without extension)
        '''

        import matplotlib.pyplot as plt
        if not self.WoE:
            self.calc_WoE()
        if not other.WoE:
            other.calc_WoE()
        seta=set(self.WoE.keys())
        setb=set(other.WoE.keys())
        tot_set=seta.union(setb)
        
        fig=plt.figure(figsize=(6,8))
        fig.subplots_adjust(bottom=0.5)
        
        plt.xticks(rotation=90)
        labela, labelb=0,0
        tot_list=list(tot_set)
        new_list=[x for y, x in sorted([(b, a) for a, b in zip(tot_list, [float(i.split('-->')[0]) for i in tot_list])])]
        for key in new_list:
            if key in self.WoE.keys():
                v1 = self.WoE[key]
                if labela>0 :
                    plt.plot(key, v1[2],'o',fillstyle='none', c='b')
                else:                    
                    plt.plot(key, v1[2],'o',fillstyle='none', c='b', label=self.var_name)
                    labela=1
            if key in other.WoE.keys():
                v2 = other.WoE[key]
                if labelb>0 :
                    plt.plot(key, v2[2],'.',fillstyle='none', c='r')
                else:                    
                    plt.plot(key, v2[2],'.',fillstyle='none', c='r', label=other.var_name)
                    labelb=1
                    
        plt.xlabel('Groups')
        plt.ylabel('Weight of evidence')
        plt.title('Weight of evidence for the variable {}'.format(self.var_name))
        plt.legend()
        plt.show()
        if save:
            fig.savefig(name+'.png')


class observation(object):
    """
    class object observation : it is the unit used for this autogrouping project
    syntax: My_observation= observation([value, good, bad])
        value is the value of the variable for this set of observation
        good is the number of observation that are classified as good
        bad is the number of observation that are classified as bad
    """

    def __init__(self, values):
        self.value = values[0]
        self.good = values[1]
        self.bad = values[2]
        self.bad_rate = round(self.bad/(self.good+self.bad), 2)

    def __str__(self):
        return self.value

class listobs(object):

    def __init__(self):
        self.dic = {}

    def addobs(self, obs):
        self.dic[obs.value] = (obs.good, obs.bad, obs.bad_rate)

    def __str__(self):
        print('Value\tGood\tBad\tBad Rate')
        for k, v in self.dic.items():
            print(k, ': \t', v[0], ' \t', v[1], '\t', v[2])
        return str()

    #legacy code
#    def getobs(self, key):
#        return [key, self.dic[key][0], self.dic[key][1]]

class Groupobservation(listobs):


    def label(self):
        return str(min(self.dic.keys()))+'-->'+str(max(self.dic.keys()))

def autogroup(dataframe, variable, flag, def_value=None,group_method='Paragon', number_bin=30, p_value=0.05,target=8, rounding=2, w=None):
    """
    return a grouping object.
    the key words are:
    dataframe       : name of the dataframe
    variable        : name of the variable (or column) to group (format: "my_variable")
    flag            : name of the variable you are trying to predict (for example : 'bad_flag')
    def_value       : a list containing the value that we want to treat as categorical 
    group_method    : method used to create the grouping it can take 2 values:
           'Paragon'    : it is the method used by the Paragon software. This method tends to under-fit the data
           'IV'         : this method has been design to reduce the loss of information value. This method tends to over-fit the data
    number_bin      : number of bin that are initially created by the algorithm before the grouping
    p_value         : threshold used to determine if two groups are different or not
    target          : only available with the 'IV' group method and it determines the minimal number of group that will be created by the algorithm
    rounding        : as its name suggest is a rounding that is applied to the data before grouping. It increases speed, reduces over fitting and create shorter name for the group
    w               : name of the variable containing the weight to apply for each observation

    Note: for categorical variables one should use the function autogroupcat
    """

    import pandas as pd
    import scipy.stats as ss
    
    
    def paragon_group(list_of_bin, p_value):
        grouped=[]
        while len(list_of_bin) > 1:
            cum_good_0 = list_of_bin[0][1].tolist()
            cum_bad_0 = list_of_bin[0][0].tolist()
        
            cum_tot_0 = []
            for i in range(len(cum_good_0)):
                cum_tot_0 += [0]*round(cum_good_0[i]+0.49)
                cum_tot_0 += [1]*round(cum_bad_0[i]+0.49)
        
        
            cum_good_1 = list_of_bin[1][1].tolist()
            cum_bad_1 = list_of_bin[1][0].tolist()
        
            cum_tot_1 = []
        
            for i in range(len(cum_good_1)):
                cum_tot_1 += [0]*round(cum_good_1[i]+0.49)
                cum_tot_1 += [1]*round(cum_bad_1[i]+0.49)
            if ss.ttest_ind(cum_tot_0, cum_tot_1)[1] > p_value:
        
                list_of_bin[0] = list_of_bin[0].append(list_of_bin[1])
                del list_of_bin[1]
            else:
                grouped+=[list_of_bin.pop(0)]
        
        grouped+=[list_of_bin.pop(0)]
        return grouped
    
    
    def group_IV(list_of_bin, p_value, threshold):
        bad=[]
        good=[]
        for datafr in list_of_bin:
            bad+=[datafr[1].sum()]
            good+=[datafr[0].sum()]
        
        bad=np.array(bad).clip(min=1)
        good=np.array(good).clip(min=1)
        
        
        Zs=abs(ss.norm.ppf(p_value))
     
        stopcondition=False
        while len(bad)>threshold and stopcondition==False:
            pcgood=good/sum(good)
            pcbad=bad/sum(bad)
        
            WoE=np.log(pcgood/pcbad)
            IV=(pcgood-pcbad)*WoE
            BR=bad/(bad+good)
            IVgrouped=[]
            IVsep=[]
            LL=[]
            diffBR=[]
            SD=BR*(1-BR)/(good+bad)
            for i in range(len(good)-1):
                Sg=pcgood[i]+pcgood[i+1]
                Sb=pcbad[i]+pcbad[i+1]
                IVgrouped+=[(Sg-Sb)*np.log(Sg/Sb)]
                IVsep+=[IV[i]+IV[i+1]]
                diffBR+=[abs(BR[i]-BR[i+1])  ]  
                LL+=[diffBR[-1]-Zs*np.sqrt(SD[i]+SD[i+1])]
            diffBR=np.array(diffBR)
            k=diffBR.argmin()
            
            if LL[k]<0:
                list_of_bin[k]=list_of_bin[k].append(list_of_bin.pop(k+1))
                bad=[]
                good=[]
                for datafr in list_of_bin:
                    bad+=[datafr[1].sum()]
                    good+=[datafr[0].sum()]
                
                bad=np.array(bad).clip(min=1)
                good=np.array(good).clip(min=1)
            else:
                stopcondition=True
        return list_of_bin
    
    
    name_group = grouping(dataframe, variable, dataframe[variable].tolist())
    name_group.defval=def_value
    df=dataframe.copy()
    
    if def_value != None:
        for elem in def_value:
            if str(elem) in set(df[variable].astype('str').tolist()):
                subtables = df[df[variable].astype('str') == str(elem)]
                subtable = pd.crosstab(subtables[variable], subtables[flag],margins=True)
                subtable['value'] = subtable.index.tolist()
                 
                Go = Groupobservation()
                good = subtable[subtable['value'].astype('str') == str(elem)][0].values[0]
                bad = subtable[subtable['value'].astype('str') == str(elem)][1].values[0]
                if w:
                    goodw = subtables[subtables[flag]==0][w].mean()
                    badw = subtables[subtables[flag]==1][w].mean()
                else:
                    goodw = 1
                    badw = 1
                Go.addobs(observation([str(elem), good*goodw, bad*badw]))
                name_group.addGroup(Go)
                df = df[df[variable].astype('str') != str(elem)]
    if len(df)==0:
        return name_group


    df['togroup'] = round(df[variable], rounding)
    table = pd.crosstab(df.togroup, df[flag], margins=True)
    table['value'] = table.index.tolist()
    table = table.reset_index(drop=True)
    table = table[table.value != 'All']
    if w:
        tabl0 = df[df[flag] == 0].groupby(['togroup'])[w].mean()
        tabl1 = df[df[flag] == 1].groupby(['togroup'])[w].mean()
        table = table.join(tabl0, on='value')
        table['w0'] = table[w]
        table = table.drop([w], axis=1)
        table = table.join(tabl1, on='value')
        table['w1'] = table[w]
        table = table.drop([w], axis=1)
        table = table.fillna(0)
        table[0] = table[0]*table.w0
        table[1] = table[1]*table.w1



#    if def_value != None:
#        for elem in def_value:
#            subtable = table[table.value == elem]
#            if len(subtable > 0):
#                Go = Groupobservation()
#                good = subtable[subtable['value'] == elem][0].values[0]
#                bad = subtable[subtable['value'] == elem][1].values[0]
#                Go.addobs(observation([elem, good, bad]))
#                name_group.addGroup(Go)
#                table = table[table.value != elem]


    table = table.reset_index(drop=True)



    list_of_bin = []
    index = 0

    while index < len(table):
        sumi  = 0

        a = table[table.index == index]
        sumi = a.All.sum()
        while sumi < len(dataframe)/number_bin and index < len(table):
            index += 1
            a = a.append(table[table.index == index])
            sumi = a.All.sum()
        index += 1
        list_of_bin += [a]
        
    if group_method=='Paragon':
        list_of_bin=paragon_group(list_of_bin, p_value)
    elif group_method=='IV':
        list_of_bin=group_IV(list_of_bin, p_value, target)
    else:
        raise ValueError('Unkown method of grouping')


            
    for bins in list_of_bin:
        Go = Groupobservation()
        for obs in bins.iterrows():
            Go.addobs(observation([obs[1]['value'], obs[1][0], obs[1][1]]))
        name_group.addGroup(Go)
    
    name_group.calc_WoE()
    
    return name_group


class Groupcategory(listobs):


    def label(self):
        name=''
        for key in sorted(self.dic.keys()):
            name+=str(key)+', '
        name=name[:-2]
        if len(name)>32:
            name=name[:15]+'...'+name[-15:]
        return name

class groupcat(grouping):
    
    def renamegroup(self, name):
        if len(self.dic[name]) == 0:
            del self.dic[name]
            return
        label = ''
        for key in sorted(self.dic[name]):
            label+=str(key)+', '
        label=label[:-2]
        if len(label)>32:
            label=label[:15]+'...'+label[-15:]
        
        self.dic[label] = self.dic[name]
        if name != label:
            del self.dic[name]

    def __str__(self):

        self.calc_WoE()
        #calculate the total number of good and bad in order to calculate the volume
        totgood = 0
        totbad = 0
        for k, v in self.dic.items():
            for ka, va in self.dic[k].items():
                totgood += va[0]
                totbad += va[1]


        print('{:<40s}{:>8s}{:>8s}{:>10s}{:>12s}{:>8s}'.format('Value','Good','Bad','Vol','Bad Rate', 'WoE'))
        #sort the different group in alphabetic order then print the different characteristics
        #for each group
        for k1 in sorted( self.dic.keys()):
            v1 = self.dic[k1]
            bad = 0
            good = 0
            for k2, v2 in v1.items():
                bad += v2[1]
                good += v2[0]
            goody=str(int(good))
            bady=str(int(bad))
            vol=str(round(100*(good+bad)/(totgood+totbad),1))+' %'
            BR=str(round(bad/(bad+good), 2))
            woe=str( round(self.WoE[k1][2],2))
            print('{:<40s}{:>8s}{:>8s}{:>10s}{:>12s}{:>8s}'.format(k1+':',goody,bady,vol,BR,woe))
        print('Information value: {}'.format(round(self.IV, 2)))
        print('Gini: {}'.format(round(self.gini, 2)))
        return str()
    
    def printdetail(self):
        '''
        print the different values in each group and their characteristics
        '''
        #sort the different group in aphabetic order then, for each group,  print the different characteristics for each observation
        for k1 in sorted( self.dic.keys()):
            v1 = self.dic[k1]
            print('\ngroup ', k1, ':')
            print('Value\tgood\tbad\tbad rate')
            bad = 0
            good = 0


            for k2 in sorted(v1):
                v2 = self.dic[k1][k2]
                bad += v2[1]
                good += v2[0]
                print(k2, ':\t', round(v2[0]), '\t', round(v2[1]), '\t', round(v2[1]/(v2[1]+v2[0]), 2))
            print('total:\t', round(good), '\t', round(bad), '\t', round(bad/(bad+good), 2))
        return str()

    def movegroup(self, src, dest=None, obs=None):
        '''
        This method takes 3 arguments:
            src :the group from where the observations come
            dest:the group where the observations should be merged in 
                    (if omitted, a new group will be created)
            obs : list of the obervations to move
        '''
        if obs.__class__ != list:
            raise ValueError('observations to move should be in a list')
        if dest == None:
            self.dic['temp'] = {}
            dest = 'temp'

        for value in obs:
            _temp = self.dic[src].pop(value)
            self.dic[dest].update({value:_temp})
        self.renamegroup(dest)
        self.renamegroup(src)

    def transform(self, path, calc=True, group=False):
        '''
        Method that matches the different values in the dataset used for the autogrouping and
        store an ordered list of weight of evidence in the dataframe mentionned by path.
        The method that take 3 arguments:
            path:    name of the dataset where the WoE should be store
            calc:    bolean indicating if the WoE should be calculated before mapping it to the data
                     calc=False allows to use WoE that has been imported
                     the default value is True
            group:   boolean indicating if a variable storing the group to which the observation belong should be created
        '''
        #calculate the Weight of Evidence if needed
        if calc == True:
            self.calc_WoE()

        #raise an error in case where no Weight of evidence has been calculated
        elif len(self.WoE) == 0:
            raise ValueError('Weight of evidence should be calculted')

        #create the name of the new variable and the list to store the values
        name = self.var_name+'_WoE'
        transname = []

        
        valdic={}
        for keys in self.WoE.keys():
            list_of_elem=list(self.dic[keys].keys())
            for elem in list_of_elem:
                valdic[elem]=[self.WoE[keys][2],keys]
        to_append=[]
        for val in self.values:
            to_append+=[valdic[str(val)][0]]
            transname+=[valdic[str(val)][1]]
        path[name]=to_append
        
        if group==True:
            namey=self.var_name+'_grp'
            path[namey] = transname


    def apply(self, src, dest, calc=True, group=False):
        '''
        Method that match the different values in a dataset specify by src and
        store an ordered list of weight of evidence (calculated on the autogrouping) in the dataframe mentionned by dest.
        The method that take 4 arguments:
            src:     specification of the value that should be match to the Weight of Evidence calculated
                     Format expected: DataFrame['name_of_the_variable']
            dest:    name of the dataset where the WoE should be store
            calc:    bolean indicating if the WoE should be calculated before mapping it to the data
                     calc=False allows to use WoE that has been imported
                     the default value is True
            group:   boolean indicating if a variable storing the group to which the observation belong should be created
        Note the value used to calculate the Weight of Evidence are the Values used when creating the object and not the one specified in the method
        '''

        #calculate the Weight of Evidence if needed
        if calc == True:
            self.calc_WoE()

        #raise an error in case where no Weight of evidence has been calculated
        elif len(self.WoE) == 0:
            raise ValueError('Weight of evidence should be calculted')

        #create the name of the new variable and the list to store the values
        name = self.var_name+'_WoE'
        
        transname=[]
        valdic={}
        for keys in self.WoE.keys():
            list_of_elem=list(self.dic[keys].keys())
            for elem in list_of_elem:
                 valdic[elem]=[self.WoE[keys][2],keys]

        to_append=[]
        for val in src:
            to_append+=[valdic[str(val)][0]]
            transname+=[valdic[str(val)][1]]
            
        dest[name]=to_append    
        if group==True:
            namey=self.var_name+'_grp'
            dest[namey] = transname          
        

    def graph(self, calc=True, save=False, name='group'):
        '''
        Method that creates a graph with the Weight of Evidence for the different groups
        The method takes three arguments:
            calc:    bolean indicating if the WoE should be calculated before mapping it to the data
                     calc=False allows to use WoE that has been imported
                     the default value is True
            save:    bolean indicating if the graph generated should be saved
            name:    string containing the name of the file (without extension)
        '''
        import matplotlib.pyplot as plt
        #calculate the Weight of Evidence if needed
        if calc == True:
            self.calc_WoE()

        #raise an error in case where no Weight of evidence has been calculated
        elif len(self.WoE) == 0:
            raise ValueError('Weight of evidence should be calculted')

        #create the graph
        fig=plt.figure(figsize=(6,8))
        fig.subplots_adjust(bottom=0.5)

        plt.xticks(rotation=90)
        #for each group in a list in alphabetic oder, plot the Weight of Evidence
        for k1 in sorted( self.dic.keys()):
            v1 = self.WoE[k1]
            plt.plot(k1, v1[2], '.', c='r')
        plt.xlabel('Groups')
        plt.ylabel('Weight of evidence')
        plt.title('Weight of evidence for the variable {}'.format(self.var_name))
        plt.show()
        if save:
            fig.savefig(name+'.png')


def autogroupcat(dataframe, variable, flag, def_value=None,group_method='Paragon', number_bin=30, p_value=0.05,target=8, w=None):
    """
    return a grouping object.
    the key words are:
    dataframe       : name of the dataframe
    variable        : name of the variable (or column) to group (format: "my_variable")
    flag            : name of the variable you are trying to predict (for example : 'bad_flag')
    def_value       : a list containing the value that we want to treat as categorical 
    group_method    : method used to create the grouping it can take 2 values:
           'Paragon'    : it is the method used by the Paragon software. This method tends to under-fit the data
           'IV'         : this method has been design to reduce the loss of information value. This method tends to over-fit the data
    number_bin      : number of bin that are initially created by the algorithm before the grouping
    p_value         : threshold used to determine if two groups are different or not
    target          : only available with the 'IV' group method and it determines the minimal number of group that will be created by the algorithm
    rounding        : as its name suggest is a rounding that is applied to the data before grouping. It increases speed, reduces over fitting and create shorter name for the group
    w               : name of the variable containing the weight to apply for each observation

    Note: for categorical variables one should use the function autogroupcat
    """

    import pandas as pd
    import scipy.stats as ss
    
    
    def paragon_group(list_of_bin, p_value):
        grouped=[]
        while len(list_of_bin) > 1:
            cum_good_0 = list_of_bin[0][1].tolist()
            cum_bad_0 = list_of_bin[0][0].tolist()
        
            cum_tot_0 = []
            for i in range(len(cum_good_0)):
                cum_tot_0 += [0]*round(cum_good_0[i]+0.49)
                cum_tot_0 += [1]*round(cum_bad_0[i]+0.49)
        
        
            cum_good_1 = list_of_bin[1][1].tolist()
            cum_bad_1 = list_of_bin[1][0].tolist()
        
            cum_tot_1 = []
        
            for i in range(len(cum_good_1)):
                cum_tot_1 += [0]*round(cum_good_1[i]+0.49)
                cum_tot_1 += [1]*round(cum_bad_1[i]+0.49)
            if ss.ttest_ind(cum_tot_0, cum_tot_1)[1] > p_value:
        
                list_of_bin[0] = list_of_bin[0].append(list_of_bin[1])
                del list_of_bin[1]
            else:
                grouped+=[list_of_bin.pop(0)]
        
        grouped+=[list_of_bin.pop(0)]
        return grouped
    
    
    def group_IV(list_of_bin, p_value, threshold):
        bad=[]
        good=[]
        for datafr in list_of_bin:
            bad+=[datafr[1].sum()]
            good+=[datafr[0].sum()]
        
        bad=np.array(bad).clip(min=1)
        good=np.array(good).clip(min=1)
        
        
        Zs=abs(ss.norm.ppf(p_value))
     
        stopcondition=False
        while len(bad)>threshold and stopcondition==False:
            pcgood=good/sum(good)
            pcbad=bad/sum(bad)
        
            WoE=np.log(pcgood/pcbad)
            IV=(pcgood-pcbad)*WoE
            BR=bad/(bad+good)
            IVgrouped=[]
            IVsep=[]
            LL=[]
            diffBR=[]
            SD=BR*(1-BR)/(good+bad)
            for i in range(len(good)-1):
                Sg=pcgood[i]+pcgood[i+1]
                Sb=pcbad[i]+pcbad[i+1]
                IVgrouped+=[(Sg-Sb)*np.log(Sg/Sb)]
                IVsep+=[IV[i]+IV[i+1]]
                diffBR+=[abs(BR[i]-BR[i+1])  ]  
                LL+=[diffBR[-1]-Zs*np.sqrt(SD[i]+SD[i+1])]
            diffBR=np.array(diffBR)
            k=diffBR.argmin()
            
            if LL[k]<0:
                list_of_bin[k]=list_of_bin[k].append(list_of_bin.pop(k+1))
                bad=[]
                good=[]
                for datafr in list_of_bin:
                    bad+=[datafr[1].sum()]
                    good+=[datafr[0].sum()]
                
                bad=np.array(bad).clip(min=1)
                good=np.array(good).clip(min=1)
            else:
                stopcondition=True
        return list_of_bin
    
    
    name_group = groupcat(dataframe, variable, dataframe[variable].tolist())
    name_group.defval=def_value
    
    df=dataframe.copy()
    if def_value != None:
        for elem in def_value:
            if str(elem) in set(df[variable].astype('str').tolist()):
                subtables = df[df[variable].astype('str') == str(elem)]
                subtable = pd.crosstab(subtables[variable], subtables[flag],margins=True)
                subtable['value'] = subtable.index.tolist()
    
                
                Go = Groupcategory()
                good = subtable[subtable['value'].astype('str') == str(elem)][0].values[0]
                bad = subtable[subtable['value'].astype('str') == str(elem)][1].values[0]
                if w:
                    goodw = subtables[subtables[flag]==0][w].mean()
                    badw = subtables[subtables[flag]==1][w].mean()
                else:
                    goodw = 1
                    badw = 1
                Go.addobs(observation([str(elem), good*goodw, bad*badw]))
                name_group.addGroup(Go)
                df = df[df[variable].astype('str') != str(elem)]

    if len(df)==0:
        return name_group
    
    table = pd.crosstab(df[variable], df[flag], margins=True)
    table['value'] = table.index.tolist()
    table = table.reset_index(drop=True)
    table = table[table.value != 'All']
    if w:
        tabl0 = df[df[flag] == 0].groupby([variable])[w].mean()
        tabl1 = df[df[flag] == 1].groupby([variable])[w].mean()
        table = table.join(tabl0, on='value')
        table['w0'] = table[w]
        table = table.drop([w], axis=1)
        table = table.join(tabl1, on='value')
        table['w1'] = table[w]
        table = table.drop([w], axis=1)
        table = table.fillna(0)
        table[0] = table[0]*table.w0
        table[1] = table[1]*table.w1
        
    table['BR'] = table[1]/(table[0]+table[1])
    table = table.sort_values(['BR'])


    table = table.reset_index(drop=True)



    list_of_bin = []
    index = 0

    while index < len(table):
        sumi  = 0

        a = table[table.index == index]
        sumi = a.All.sum()
        while sumi < len(dataframe)/number_bin and index < len(table):
            index += 1
            a = a.append(table[table.index == index])
            sumi = a.All.sum()
        index += 1
        list_of_bin += [a]
        
    if group_method=='Paragon':
        list_of_bin=paragon_group(list_of_bin, p_value)
    elif group_method=='IV':
        list_of_bin=group_IV(list_of_bin, p_value, target)
    else:
        raise ValueError('Unkown method of grouping')


            
    for bins in list_of_bin:
        Go = Groupcategory()
        for obs in bins.iterrows():
            Go.addobs(observation([str(obs[1]['value']), obs[1][0], obs[1][1]]))
        name_group.addGroup(Go)
        
    name_group.calc_WoE()
    return name_group    