#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
df=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df


# In[2]:


df.head()


# In[3]:


df.tail()


# In[5]:


import pandas as pd 
df=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12])
df


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,5))
plt.title("Total orders for each discount")
sns.countplot(df['Discount'],label="Count")


# In[13]:


plt.figure(figsize=(20,5))
plt.title("Discount wise sales")
sns.barplot(x = df['Discount'] , y = df["Sales"])


# In[14]:


plt.figure(figsize=(20,5))
plt.title("Effect of discount on Profit")
sns.barplot(x = df['Discount'] , y = df["Profit"])


# In[10]:


plt.figure(figsize=(20,5))
sns.heatmap(df.corr(), cmap = 'Blues', annot = True)
plt.show()


# In[9]:


state_CA = "California"

group_state = df.groupby(['State'])


category = group_state.get_group(state_CA).groupby(df['Category'])

fig, ax = plt.subplots(1, 3, figsize = (30,5))
fig.suptitle("California",fontweight='bold',fontsize=20)        
ax_index = 0

for cat in ['Furniture', 'Office Supplies', 'Technology']:
            category_data = category.get_group(cat).groupby(['Sub-Category']).sum()
            sns.barplot(x=category_data.index  , y = category_data.Profit, ax = ax[ax_index])
            ax[ax_index].set_xlabel(cat)
            ax_index +=1

plt.show()


# In[10]:


state_CA = "New York"

group_state = df.groupby(['State'])


category = group_state.get_group(state_CA).groupby(df['Category'])

fig, ax = plt.subplots(1, 3, figsize = (30,5))
fig.suptitle("New York",fontweight='bold',fontsize=20)        
ax_index = 0

for cat in ['Furniture', 'Office Supplies', 'Technology']:
            category_data = category.get_group(cat).groupby(['Sub-Category']).sum()
            sns.barplot(x=category_data.index  , y = category_data.Profit, ax = ax[ax_index])
            ax[ax_index].set_xlabel(cat)
            ax_index +=1

plt.show()


# In[11]:


#19. Texas market Share
state_CA = "Texas"

group_state = df.groupby(['State'])


category = group_state.get_group(state_CA).groupby(df['Category'])

fig, ax = plt.subplots(1, 3, figsize = (30,5))
fig.suptitle("Texas",fontweight='bold',fontsize=20)        
ax_index = 0

for cat in ['Furniture', 'Office Supplies', 'Technology']:
            category_data = category.get_group(cat).groupby(['Sub-Category']).sum()
            sns.barplot(x=category_data.index  , y = category_data.Profit, ax = ax[ax_index])
            ax[ax_index].set_xlabel(cat)
            ax_index +=1

plt.show()


# In[12]:


state_CA = "Pennsylvania"

group_state = df.groupby(['State'])


category = group_state.get_group(state_CA).groupby(df['Category'])

fig, ax = plt.subplots(1, 3, figsize = (30,5))
fig.suptitle("Pennsylvania",fontweight='bold',fontsize=20)        
ax_index = 0

for cat in ['Furniture', 'Office Supplies', 'Technology']:
            category_data = category.get_group(cat).groupby(['Sub-Category']).sum()
            sns.barplot(x=category_data.index  , y = category_data.Profit, ax = ax[ax_index])
            ax[ax_index].set_xlabel(cat)
            ax_index +=1

plt.show()


# In[13]:


attributes=['Sales','Quantity','Discount','Profit']
corr_mat=df.corr()
corr_mat


# In[14]:


plt.subplots(figsize=(15,7))
sns.heatmap(corr_mat,annot=True)
plt.show()


# In[15]:


sns.pairplot(df,hue='Region',diag_kind='hist')


# In[ ]:


plt.figure(figsize=(15,7))
sns.boxplot(df['Sales'])
plt.show()


# In[ ]:


plt.figure(figsize=(15,7))
sns.boxplot(df['Quantity'])
plt.show()


# In[ ]:


plt.figure(figsize=(15,7))
sns.boxplot(df['Discount'])
plt.show()


# In[16]:


plt.figure(figsize=(15,7))
sns.boxplot(df['Profit'])
plt.show()


# In[17]:


plt.figure(figsize=(15,10))
sns.barplot(x='Sub-Category',y='Sales',data=df)
plt.show()


# In[18]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[19]:


plt.figure(figsize = (15, 7))
# plot Sales and Profit for comparisons
sns.kdeplot(df['Sales'], color = 'Teal', label = 'Sales', shade = True, bw = 25)
sns.kdeplot(df['Profit'], color = 'Red', label = 'Profit', shade = True, bw = 25)
#plt.xlim([0, 13000])
#plt.ylim([0, 0.00007])
plt.ylabel('Density')
plt.xlabel('Monetary Value in USD$')
plt.title('Sales and Profit', fontsize = 20)
plt.legend(loc = 'upper right', frameon = False) 
plt.show()


# In[20]:


plt.figure(figsize = (15, 7))
# plot Sales and Profit for comparisons
sns.kdeplot(df['Sales'], color = 'Teal', label = 'Sales', shade = True, bw = 25)
sns.kdeplot(df['Profit'], color = 'Red', label = 'Profit', shade = True, bw = 25)
plt.xlim([0, 13000])
plt.ylim([0, 0.00007])
plt.ylabel('Density')
plt.xlabel('Monetary Value in USD$')
plt.title('Sales and Profit', fontsize = 20)
plt.legend(loc = 'upper right', frameon = False) 
plt.show()


# In[21]:


sns.distplot(df['Discount'], color='red')


# In[22]:


from ipywidgets import widgets,interact
drop_down_values=widgets.Dropdown(options=['Profit','Discount','Sales','Quantity'],value='Quantity')
drop_down_agg=widgets.Dropdown(options=['sum','mean','std','max','min'])
drop_down_var1=widgets.Dropdown(options=list(df.describe(include='object').columns),value='City')
drop_down_var2=widgets.Dropdown(options=list(df.describe(include='object').columns),value='Sub-Category')


def crosstab(var1,var2,values,agg):
    return pd.crosstab(df[var1],df[var2],margins=True,values=df[values],aggfunc=agg)

interact(crosstab,var1=drop_down_var1,var2=drop_down_var2,values=drop_down_values,agg=drop_down_agg)


# In[23]:


drop_down=widgets.Dropdown(options=list(df['Sub-Category'].unique()))

def graph(sub_category):
    sns.histplot(df[df['Sub-Category']==sub_category].Profit/df[df['Sub-Category']==sub_category].Quantity)

interact(graph,sub_category=drop_down)


# In[24]:


import numpy as np
profit=df[df.Profit>0]
loss=df[df.Profit<0]

# percentage share of total profit by each sub-category
plt.pie(profit.groupby('Sub-Category').agg('sum').Profit,radius=3.12,labels=profit.groupby('Sub-Category').agg('sum').index,
       autopct='%1.2f%%')
plt.title('Profit pie',fontdict=dict(fontsize=36),pad=100,loc='center')
plt.show()

# percentage share of total loss by each sub-category
plt.pie(np.abs(loss.groupby('Sub-Category').agg('sum').Profit),radius=3.12,labels=loss.groupby('Sub-Category').agg('sum').index,
       autopct='%1.2f%%')
plt.title('Loss pie',fontdict=dict(fontsize=36),pad=100,loc='center')
plt.show()


# In[25]:


sns.set(style="whitegrid")
plt.figure(2, figsize=(20,15))
sns.barplot(x='Sub-Category',y='Profit', data=df, palette='Spectral')
plt.suptitle('Pie Consumption Patterns in the United States', fontsize=16)
plt.show()


# In[26]:


plt.figure(figsize=(10,4))
sns.lineplot('Discount','Profit', data=df , color='y',label='Discount')
plt.legend()
plt.show()


# In[27]:


df['Sales'].max()


# In[28]:


df['Sales'].min()


# In[29]:


df['Sales'].sum()


# In[30]:


df['Sales'].mean()


# In[31]:


df.isnull().sum()


# In[32]:


df.mean()


# In[33]:


df.max()


# In[34]:


df.min()


# In[35]:


df.duplicated().sum()


# In[36]:


df.corr()


# In[37]:


df.cov()


# In[38]:


df.hist(bins=50,figsize=(15,7))
plt.show()


# In[39]:


df['City'].unique()


# In[40]:


cities_profit = pd.DataFrame(df.groupby('City')['Profit'].sum())
cities_profit.reset_index(inplace=True)
cities_profit = cities_profit.sort_values( by="Profit",ascending=False)
cities_profit.head(10)


# In[41]:


import plotly.express as px
fig = px.treemap(cities_profit, 
                 path=['City' ,'Profit'], 
                 color_continuous_scale='deep',
                 values='Profit',color='Profit')
fig.update_layout(width=1000,height=500)
fig.show()


# In[42]:


df['State'].unique()


# In[43]:


states_profit = pd.DataFrame(df.groupby('State')['Profit'].sum())
states_profit.reset_index(inplace=True)
states_profit = states_profit.sort_values( by="Profit",ascending=False)
states_profit.head(10)


# In[44]:


import plotly.express as px
fig = px.treemap(states_profit, 
                 path=['State' ,'Profit'], 
                 color_continuous_scale='deep',
                 values='Profit',color='Profit')
fig.update_layout(width=1000, height=500)
fig.show()


# In[45]:


#to find the total sales
print("Total Sales:" + str(round(sum(df['Sales']))))
#total quantity sold
print("Total Quantity:" + str(sum(df['Quantity'])))
#total profit
print("Total Profit:" + str(round(sum(df['Profit']))))


# In[46]:


import seaborn as sns
sns.countplot(x=df['Ship Mode'])


# In[47]:


import matplotlib.pyplot as plt
df.groupby('Ship Mode')[['Profit','Sales']].sum().plot.bar(color=['royalblue','chocolate'])
plt.ylabel('Sales')


# In[48]:


tp = {'fontsize': 12}
plt.title("Category")
#piechart is in matplotlib
plt.pie(df['Category'].value_counts(), labels = df['Category'].value_counts().index, autopct='%1.2f%%', textprops=tp)
plt.show()


# In[49]:



df.groupby('Category')[['Profit', 'Sales']].sum().plot.bar(color=['royalblue', 'chocolate'], figsize=(15, 7))


# In[50]:


statewise=df.groupby(['Sub-Category'])['Profit'].sum().nlargest(10)
statewise.plot.barh(figsize=(10,8))
plt.xlabel('Profit($)')


# In[51]:


plt.figure(figsize=(18,10))
sns.countplot(x="State", data=df, palette='Paired', order=df['State'].value_counts().index)
plt.xticks(rotation=90)
plt.ylabel('Number of orders')
plt.show()


# In[52]:


import matplotlib.pyplot as plt
df.groupby('Segment')[['Profit','Sales']].sum().plot.bar(color=['chocolate','royalblue'], figsize=(8,5))
plt.ylabel('Profit/Loss and sales')
plt.show()


# In[53]:


plt.figure(figsize=(8,5))
sns.countplot(x='Region',hue='Ship Mode',data=df)
plt.ylabel('Count of Ship Mode')
plt.title('Count of ship mode by region and ship mode')


# In[55]:


df.groupby('Region')[['Profit','Sales']].sum().plot.bar(color=['royalblue','chocolate'], figsize=(8,5))
plt.ylabel('Profit & sale in ($)')
plt.show()


# In[56]:


df = pd.DataFrame.drop_duplicates(df)
df.shape


# In[57]:


df = df.drop(['Country'], axis = 1)
df = df.drop(['Postal Code'], axis = 1)


# In[58]:


df.nunique()


# In[59]:


category_analysis=pd.DataFrame(df.groupby(['Category'])[['Sales','Profit','Quantity']].sum())
category_analysis


# In[61]:


sns.set_style('whitegrid')
figure,axis=plt.subplots(1,3,figsize=(8,5))
cat1= sns.barplot(x = category_analysis.index, y = category_analysis.Sales, ax=axis[0] )
cat2= sns.barplot(x = category_analysis.index, y = category_analysis.Profit, ax=axis[1] )
cat3= sns.barplot(x = category_analysis.index, y = category_analysis.Quantity, ax=axis[2] )
cat1.set(title = 'Sales')
cat2.set(title = 'Profit')
cat3.set(title = 'Quantity')
plt.setp(cat1.get_xticklabels(), rotation = 'vertical', size=9)
plt.setp(cat2.get_xticklabels(), rotation = 'vertical', size=9)
plt.setp(cat3.get_xticklabels(), rotation = 'vertical', size=9)
figure.tight_layout()


# In[62]:


plt.figure(figsize=(12,6))
sns.set_theme(style="white")
corr = df.corr()
heatmap = sns.heatmap(corr, annot=True, cmap = 'GnBu')


# In[63]:


df = pd.DataFrame(df.sort_values('Profit', ascending = False))

df


# In[64]:


sns.set_theme(style="whitegrid")


figure, axis = plt.subplots(1, 2, figsize=(12, 6))

subcat1 = sns.barplot(data = df, x = df.index, y = df.Sales, ax=axis[0], palette = "rainbow")
subcat1.set(title="Best Selling Sub-Category")
subcat1.set_xticklabels(subcat1.get_xticklabels(),rotation = "vertical", size = 10)

subcat2 = sns.barplot(data = df, x = df.index, y = df.Profit, ax=axis[1], palette = "coolwarm")
subcat2.set(title = "Most Profitable Sub-Category")
subcat2.set_xticklabels(subcat2.get_xticklabels(),rotation = "vertical", size = 10)

figure.tight_layout()

plt.show()


# In[65]:


df['Region'].value_counts()


# In[66]:


sns.countplot(x=df['Region'])


# In[67]:


df['Sub-Category'].value_counts()


# In[68]:


plt.figure(figsize=(8,8))
plt.pie(df['Sub-Category'].value_counts(),labels=df['Sub-Category'].value_counts().index,autopct='%.2f%%')
plt.show()


# In[69]:


profits=df.groupby(['State'])['Profit'].sum().nlargest(10)
profits


# In[70]:


plt.figure(figsize=(15,7))
profits.plot.bar()


# In[71]:


plt.figure(figsize=(8,8))
profits.plot.pie( autopct ="%.2f%%")


# In[72]:


Sales1=df.groupby(['State'])['Sales'].sum().nlargest(10)
Sales1


# In[73]:


plt.figure(figsize=(15,7))
Sales1.plot.bar()


# In[74]:


plt.figure(figsize=(8,8))
Sales1.plot.pie( autopct ="%.2f%%")


# In[75]:


df.info()


# In[76]:


df_f ={
    'Discount': df['Discount'],
    'Sales': df['Sales'],
    'Profit': df['Profit']
    
}

df = pd.DataFrame(df_f, columns=['Discount','Sales','Profit'])

print(df)

corr = df.corr()
ax1 = sns.heatmap(corr, cmap="Blues", annot=True)
plt.show()


# In[77]:


df.sort_values(by='Sales',ascending=False,inplace=True)
print(df)


# In[78]:


df['Profit'].max()


# In[79]:


df['Profit'].min()


# In[80]:


df['Profit'].describe()


# In[81]:


df['Profit'].sum()


# In[82]:


df.sort_values(by='Profit',ascending=False,inplace=True)
print(df)


# In[83]:


df['Discount'].max()


# In[84]:


df['Discount'].min()


# In[85]:


df['Discount'].mean()


# In[86]:


df.sort_values(by='Discount',ascending=False,inplace=True)
print(df)


# In[87]:


import pandas as pd 
df1=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df1


# In[88]:


df1=df1.groupby(['Country']).sum()
df1


# In[89]:


df1['Sales'].plot(kind='bar',ylabel='Sales',title='Total USA Sales',figsize=(15,7),width=.25,edgecolor='black',color='y')


# In[90]:


df1['Profit'].plot(kind='bar',ylabel='Sales',title='Total USA Profit',figsize=(15,7),width=.25,edgecolor='black',color='b')


# In[91]:


df1['Profitablity_Ratio']=df1['Profit']/df1['Sales']*100
df1


# In[92]:


df1['Profitablity_Ratio'].plot(kind='bar',ylabel='Profitabilty_Ratio',title='Total USA Profitability_Ratio',figsize=(15,7),width=.25,edgecolor='black',color='g')


# In[93]:


import pandas as pd 
df_R=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_R


# In[94]:


df_R=df_R.groupby(['Region']).sum()
df_R


# In[95]:


df_R['Sales'].plot(kind='bar',ylabel='Sales',title='Region wise Sales',figsize=(15,7))


# In[96]:


df_R['Sales'].plot(kind='pie',ylabel='Sales',title='Region wise Sales',figsize=(15,7))


# In[97]:


df_R.sort_values(by='Sales',ascending=False,inplace=True)
print(df_R)


# In[98]:


df_R['Sales'].plot(kind='barh',ylabel='Sales',title='Region Wise Sales',figsize=(15,7))


# In[99]:


df_R['Sales'].plot(kind='pie',ylabel='Sales',title='Region Wise Sales',figsize=(15,7))


# In[34]:


df_R['Profit'].plot(kind='bar',ylabel='Sales',title='Region wise Profit',figsize=(15,7))


# In[35]:


df_R['Profit'].plot(kind='pie',ylabel='Sales',title='Region wise Profit',figsize=(15,7))


# In[36]:


df_R.sort_values(by='Profit',ascending=False,inplace=True)
print(df_R)


# In[37]:


df_R['Profit'].plot(kind='barh',ylabel='Profit',title='Region Wise Profit',figsize=(15,7))


# In[38]:


df_R['Profit'].plot(kind='pie',ylabel='Profit',title='Region Wise Profit',figsize=(15,7))


# In[39]:


df_R['Profitability_Ratio']=df_R['Profit']/df_R['Sales']*100
df_R


# In[40]:


df_R['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='Region Wise Profitability_Ratio',figsize=(15,7))


# In[41]:


df_R['Profitability_Ratio'].plot(kind='pie',ylabel='Profitability_Ratio',title='Region Wise Profitability_Ratio',figsize=(15,7))


# In[42]:


import pandas as pd 
df_S=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_S


# In[100]:


df_S=df_S.groupby(['State']).sum()
print(df_S)


# In[ ]:


df_S['Sales'].plot(kind='bar',ylabel='Sales',title='State Wise Sales',figsize=(15,7))


# In[ ]:


df_S['Sales'].plot(kind='pie',ylabel='Sales',title='State Wise Sales',figsize=(15,7))


# In[ ]:


df_S.sort_values(by='Sales',ascending=False,inplace=True)
df_S


# In[ ]:


df_S['Sales'].plot(kind='bar',ylabel='Sales',title='State Wise Sales',figsize=(15,7))


# In[ ]:


df_S['Sales'].plot(kind='pie',ylabel='Sales',title='State Wise Sales',figsize=(15,7))


# In[ ]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
top_5 = df_S['Sales'].iloc[:5]

#create bar chart of top 10 teams
top_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('State')
plt.title('Top 5 State wise Sales')
plt.ylabel('Sales')


# In[ ]:


import pandas as pd 
df_S=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_S


# In[ ]:


df_S=df_S.groupby(['State']).sum()
print(df_S)


# In[ ]:


df_S.sort_values(by='Sales',ascending=True,inplace=True)
df_S


# In[ ]:


df_S['Sales'].plot(kind='bar',ylabel='Sales',title='State Wise Sales',figsize=(15,7))


# In[ ]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
bottom_5 = df_S['Sales'].iloc[:5]

#create bar chart of top 10 teams
bottom_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('State')
plt.title('Bottom 5 State wise Sales')
plt.ylabel('Sales')


# In[ ]:


df_S['Profit'].plot(kind='bar',ylabel='Profit',title='State Wise Profit',figsize=(15,7))


# In[101]:


df_S.sort_values(by='Profit',ascending=False,inplace=True)
df_S


# In[102]:


df_S['Profit'].plot(kind='bar',ylabel='Profit',title='State Wise Profit',figsize=(15,7))


# In[103]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
top_5 = df_S['Profit'].iloc[:5]

#create bar chart of top 10 teams
top_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('State')
plt.title('Top5 State wise Profit')
plt.ylabel('Profit')


# In[104]:


df_S.sort_values(by='Profit',ascending=True,inplace=True)
df_S


# In[105]:


df_S['Profit'].plot(kind='bar',ylabel='Profit',title='State Wise Profit',figsize=(15,7))


# In[106]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
bottom_5 = df_S['Profit'].iloc[:5]

#create bar chart of top 10 teams
bottom_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('State')
plt.title('Bottom5 State wise Profit')
plt.ylabel('Profit')


# In[107]:


df_S=df_S.groupby(['State']).sum()
df_S


# In[108]:


df_S['Profitability_Ratio']=df_S['Profit']/df_S['Sales']*100
df_S


# In[109]:


df_S['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='State Wise Profitability_Ratio',figsize=(15,7))


# In[110]:


df_S.sort_values(by='Profitability_Ratio',ascending=False,inplace=True)
df_S


# In[111]:


df_S['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='State Wise Profitability_Ratio',figsize=(15,7))


# In[112]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
top_5 = df_S['Profitability_Ratio'].iloc[:5]

#create bar chart of top 10 teams
top_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('State')
plt.title('Top5 State wise Profitablility_Ratio')
plt.ylabel('Profitability_Ratio')


# In[113]:


df_S.sort_values(by='Profitability_Ratio',ascending=True,inplace=True)
df_S


# In[114]:


df_S['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='State Wise Profitability_Ratio',figsize=(15,7))


# In[115]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
bottom_5 = df_S['Profitability_Ratio'].iloc[:5]

#create bar chart of top 10 teams
bottom_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('State')
plt.title('Bottom5 State wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[116]:


import pandas as pd 
df_C=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_C


# In[117]:


df_C=df_C.groupby(['City']).sum()
print(df_C)


# In[118]:


df_C['Sales'].plot(kind='bar',ylabel='Sales',title='City Wise Sales',figsize=(15,7))


# In[119]:


df_C.sort_values(by='Sales',ascending=False,inplace=True)
df_C


# In[120]:


df_C['Sales'].plot(kind='bar',ylabel='Sales',title='City Wise Sales',figsize=(15,7))


# In[121]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
top_5 = df_C['Sales'].iloc[:5]

#create bar chart of top 10 teams
top_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('City')
plt.title('Top5 City wise Sales')
plt.ylabel('Sales')


# In[122]:


df_C.sort_values(by='Sales',ascending=True,inplace=True)
df_C


# In[123]:


df_C['Sales'].plot(kind='bar',ylabel='Sales',title='City Wise Sales',figsize=(15,7))


# In[124]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
bottom_5 = df_C['Sales'].iloc[:5]

#create bar chart of top 10 teams
bottom_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('City')
plt.title('Bottom5 City wise Sales')
plt.ylabel('Sales')


# In[125]:


df_C.sort_values(by='Profit',ascending=False,inplace=True)
df_C


# In[126]:


df_C['Profit'].plot(kind='bar',ylabel='Profit',title='City Wise Profit',figsize=(15,7))


# In[127]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
top_5 = df_C['Profit'].iloc[:5]

#create bar chart of top 10 teams
top_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('State')
plt.title('Top5 City wise Profit')
plt.ylabel('Profit')


# In[128]:


df_C.sort_values(by='Profit',ascending=True,inplace=True)
df_C


# In[129]:


df_C['Profit'].plot(kind='bar',ylabel='Profit',title='City Wise Profit',figsize=(15,7))


# In[130]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
bottom_5 = df_C['Profit'].iloc[:5]

#create bar chart of top 10 teams
bottom_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('State')
plt.title('Bottom5 City wise Profit')
plt.ylabel('Profit')


# In[131]:


df_C=df_C.groupby(['City']).sum()
df_C


# In[132]:


df_C['Profitability_Ratio']=df_C['Profit']/df_C['Sales']*100
df_C


# In[133]:


df_C['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='City Wise Profitability_Ratio',figsize=(15,7))


# In[134]:


df_C.sort_values(by='Profitability_Ratio',ascending=False,inplace=True)
df_C


# In[135]:


df_C['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='City Wise Profitability_Ratio',figsize=(15,7))


# In[136]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
top_5 = df_C['Profitability_Ratio'].iloc[:5]

#create bar chart of top 10 teams
top_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('City')
plt.title('Top5 City Wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[137]:


df_C.sort_values(by='Profitability_Ratio',ascending=True,inplace=True)
df_C


# In[138]:


df_C['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='City Wise Profitability_Ratio',figsize=(15,7))


# In[139]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
bottom_5 = df_C['Profitability_Ratio'].iloc[:5]

#create bar chart of top 10 teams
bottom_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('City')
plt.title('Bottom5 City wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[140]:


import pandas as pd 
df_Seg=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_Seg


# In[141]:


df_Seg=df_Seg.groupby(['Segment']).sum()
print(df_Seg)


# In[142]:


df_Seg['Sales'].plot(kind='bar',ylabel='Sales',title='Segment wise Sales',figsize=(15,7))


# In[143]:


df_Seg['Sales'].plot(kind='pie',ylabel='Sales',title='Segment wise Sales',figsize=(15,7))


# In[144]:


df_Seg.sort_values(by='Sales',ascending=False,inplace=True)
df_Seg


# In[145]:


df_Seg['Sales'].plot(kind='bar',ylabel='Sales',title='Segment wise Sales',figsize=(15,7))


# In[146]:


df_Seg['Sales'].plot(kind='pie',ylabel='Sales',title='Segment wise Sales',figsize=(15,7))


# In[147]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
top_5 = df_Seg['Sales'].iloc[:5]

#create bar chart of top 10 teams
top_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('Segment')
plt.title('Top5 Segment wise Sales')
plt.ylabel('Sales')


# In[148]:


df_Seg.sort_values(by='Sales',ascending=True,inplace=True)
df_Seg


# In[149]:


df_Seg['Sales'].plot(kind='bar',ylabel='Sales',title='Segment wise Sales',figsize=(15,7))


# In[150]:


df_Seg['Sales'].plot(kind='pie',ylabel='Sales',title='Segment wise Sales',figsize=(15,7))


# In[151]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
bottom_5 = df_Seg['Sales'].iloc[:5]

#create bar chart of top 10 teams
bottom_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('Segment')
plt.title('Bottom5 Segment wise Sales')
plt.ylabel('Sales')


# In[152]:


df_Seg.sort_values(by='Profit',ascending=False,inplace=True)
df_Seg


# In[153]:


df_Seg['Profit'].plot(kind='bar',ylabel='Profit',title='Segment wise Profit',figsize=(15,7))


# In[154]:


df_Seg['Profit'].plot(kind='pie',ylabel='Profit',title='Segment wise Profit',figsize=(15,7))


# In[155]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
top_5 = df_Seg['Profit'].iloc[:5]

#create bar chart of top 10 teams
top_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('Segment')
plt.title('Top5 Segment wise Profit')
plt.ylabel('Profit')


# In[156]:


df_Seg.sort_values(by='Profit',ascending=True,inplace=True)
df_Seg


# In[157]:


df_Seg['Profit'].plot(kind='bar',ylabel='Profit',title='Segment wise Profit',figsize=(15,7))


# In[158]:


df_Seg['Profit'].plot(kind='pie',ylabel='Profit',title='Segment wise Profit',figsize=(15,7))


# In[159]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
bottom_5 = df_Seg['Profit'].iloc[:5]

#create bar chart of top 10 teams
bottom_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('Segment')
plt.title('Bottom5 Segment wise Profit')
plt.ylabel('Profit')


# In[160]:


df_Seg=df_Seg.groupby(['Segment']).sum()
df_Seg


# In[161]:


df_Seg['Profitability_Ratio']=df_Seg['Profit']/df_Seg['Sales']*100
df_Seg


# In[162]:


df_Seg['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='Segment wise Profitability_Ratio',figsize=(15,7))


# In[163]:


df_Seg['Profitability_Ratio'].plot(kind='pie',ylabel='Profitability_Ratio',title='Segment wise Profitability_Ratio',figsize=(15,7))


# In[164]:


df_Seg.sort_values(by='Profitability_Ratio',ascending=False,inplace=True)
df_Seg


# In[165]:


df_Seg['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='Segment wise Profitability_Ratio',figsize=(15,7))


# In[166]:


df_Seg['Profitability_Ratio'].plot(kind='pie',ylabel='Profitability_Ratio',title='Segment wise Profitability_Ratio',figsize=(15,7))


# In[167]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
top_5 = df_Seg['Profitability_Ratio'].iloc[:5]

#create bar chart of top 10 teams
top_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('Segment')
plt.title('Top5 Segment wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[168]:


df_Seg.sort_values(by='Profitability_Ratio',ascending=True,inplace=True)
df_Seg


# In[169]:


df_Seg['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='Segment wise Profitability_Ratio',figsize=(15,7))


# In[170]:


df_Seg['Profitability_Ratio'].plot(kind='pie',ylabel='Profitability_Ratio',title='Segment wise Profitability_Ratio',figsize=(15,7))


# In[171]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
bottom_5 = df_Seg['Profitability_Ratio'].iloc[:5]

#create bar chart of top 10 teams
bottom_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('Segment')
plt.title('Bottom5 Segment wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[172]:


import pandas as pd 
df_Cat=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_Cat


# In[173]:


df_Cat=df_Cat.groupby(['Category']).sum()
print(df_Cat)


# In[174]:


df_Cat['Sales'].plot(kind='bar',ylabel='Sales',title='Category wise Sales',figsize=(15,7))


# In[175]:


df_Cat['Sales'].plot(kind='pie',ylabel='Sales',title='Category wise Sales',figsize=(15,7))


# In[176]:


df_Cat.sort_values(by='Sales',ascending=False,inplace=True)
df_Cat


# In[177]:


df_Cat['Sales'].plot(kind='bar',ylabel='Sales',title='Category wise Sales',figsize=(15,7))


# In[178]:


df_Cat['Sales'].plot(kind='pie',ylabel='Sales',title='Category wise Sales',figsize=(15,7))


# In[179]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
top_3 = df_Cat['Sales'].iloc[:3]

#create bar chart of top 10 teams
top_3.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('Category')
plt.title('Top3 Category wise Sales')
plt.ylabel('Sales')


# In[180]:


df_Cat.sort_values(by='Sales',ascending=True,inplace=True)
df_Cat


# In[181]:


df_Cat['Sales'].plot(kind='bar',ylabel='Sales',title='Category wise Sales',figsize=(15,7))


# In[182]:


df_Cat['Sales'].plot(kind='pie',ylabel='Sales',title='Category wise Sales',figsize=(15,7))


# In[183]:


import matplotlib.pyplot as plt

#find teams with top 10 occurrences
bottom_5 = df_Cat['Sales'].iloc[:5]

#create bar chart of top 10 teams
bottom_5.plot(kind='bar', edgecolor='black', rot=0)

#add axis labels
plt.xlabel('Category')
plt.title('Bottom5 Category wise Sales')
plt.ylabel('Sales')


# In[184]:


df_Cat.sort_values(by='Profit',ascending=False,inplace=True)
df_Cat


# In[185]:


df_Cat['Profit'].plot(kind='bar',ylabel='Profit',title='Category wise Profit',figsize=(15,7))


# In[186]:


df_Cat['Profit'].plot(kind='pie',ylabel='Profit',title='Category wise Profit',figsize=(15,7))


# In[187]:


import matplotlib.pyplot as plt
top_5=df_Cat['Profit'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('Category')
plt.title('Top5 Category wise Profit')
plt.ylabel('Profit')


# In[188]:


df_Cat.sort_values(by='Profit',ascending=True,inplace=True)
df_Cat


# In[189]:


df_Cat['Profit'].plot(kind='bar',ylabel='Profit',title='Category wise Profit',figsize=(15,7))


# In[191]:


df_Cat['Profit'].plot(kind='pie',ylabel='Profit',title='Category wise Profit',figsize=(15,7))


# In[192]:


import matplotlib.pyplot as plt
bottom_5=df_Cat['Profit'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('Category')
plt.title('Bottom5 Category wise Profit')
plt.ylabel('Profit')


# In[193]:


df_Cat=df_Cat.groupby(['Category']).sum()
print(df_Cat)


# In[194]:


df_Cat['Profitability_Ratio']=df_Cat['Profit']/df_Cat['Sales']*100
df_Cat


# In[195]:


df_Cat['Profitability_Ratio'].plot(kind='bar',ylabel='Protitability_Ratio',title='Category wise Profitability_Ratio',figsize=(15,7))


# In[196]:


df_Cat['Profitability_Ratio'].plot(kind='pie',ylabel='Protitability_Ratio',title='Category wise Profitability_Ratio',figsize=(15,7))


# In[197]:


df_Cat.sort_values(by='Profitability_Ratio',ascending=False,inplace=True)
df_Cat


# In[198]:


df_Cat['Profitability_Ratio'].plot(kind='bar',ylabel='Protitability_Ratio',title='Category wise Profitability_Ratio',figsize=(15,7))


# In[199]:


df_Cat['Profitability_Ratio'].plot(kind='pie',ylabel='Protitability_Ratio',title='Category wise Profitability_Ratio',figsize=(15,7))


# In[200]:


import matplotlib.pyplot as plt
top_5=df_Cat['Profitability_Ratio'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('Category')
plt.title('Top5 Category wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[201]:


df_Cat.sort_values(by='Profitability_Ratio',ascending=True,inplace=True)
df_Cat


# In[202]:


df_Cat['Profitability_Ratio'].plot(kind='bar',ylabel='Protitability_Ratio',title='Category wise Profitability_Ratio',figsize=(15,7))


# In[203]:


df_Cat['Profitability_Ratio'].plot(kind='pie',ylabel='Protitability_Ratio',title='Category wise Profitability_Ratio',figsize=(15,7))


# In[204]:


import matplotlib.pyplot as plt
bottom_5=df_Cat['Profitability_Ratio'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('Category')
plt.title('Bottom5 Category wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[205]:


import pandas as pd 
df_SubCat=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_SubCat


# In[206]:


df_SubCat=df_SubCat.groupby(['Sub-Category']).sum()
print(df_SubCat)


# In[207]:


df_SubCat['Sales'].plot(kind='bar',ylabel='Sales',title='Sub-Category wise Sales',figsize=(15,7))


# In[208]:


df_SubCat['Sales'].plot(kind='pie',ylabel='Sales',title='Sub-Category wise Sales',figsize=(15,7))


# In[209]:


df_SubCat.sort_values(by='Sales',ascending=False,inplace=True)
df_SubCat


# In[210]:


df_SubCat['Sales'].plot(kind='bar',ylabel='Sales',title='Sub-Category wise Sales',figsize=(15,7))


# In[211]:


df_SubCat['Sales'].plot(kind='pie',ylabel='Sales',title='Sub-Category wise Sales',figsize=(15,7))


# In[212]:


import matplotlib.pyplot as plt
top_5=df_SubCat['Sales'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('Sub-Category')
plt.title('Top5 Category wise Sales')
plt.ylabel('Sales')


# In[213]:


df_SubCat.sort_values(by='Sales',ascending=True,inplace=True)
df_SubCat


# In[214]:


df_SubCat['Sales'].plot(kind='bar',ylabel='Sales',title='Sub-Category wise Sales',figsize=(15,7))


# In[215]:


df_SubCat['Sales'].plot(kind='pie',ylabel='Sales',title='Sub-Category wise Sales',figsize=(15,7))


# In[216]:


import matplotlib.pyplot as plt
bottom_5=df_SubCat['Sales'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('Sub-Category')
plt.title('bottom5 Category wise Sales')
plt.ylabel('Sales')


# In[217]:


df_SubCat.sort_values(by='Profit',ascending=False,inplace=True)
df_SubCat


# In[218]:


df_SubCat['Profit'].plot(kind='bar',ylabel='Profit',title='Sub-Category wise Profit',figsize=(15,7))


# In[219]:


import matplotlib.pyplot as plt
top_5=df_SubCat['Profit'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('Sub-Category')
plt.title('Top5 Sub-Category wise Profit')
plt.ylabel('Profit')


# In[220]:


df_SubCat.sort_values(by='Profit',ascending=True,inplace=True)
df_SubCat


# In[221]:


df_SubCat['Profit'].plot(kind='bar',ylabel='Profit',title='Sub-Category wise Profit',figsize=(15,7))


# In[222]:


import matplotlib.pyplot as plt
bottom_5=df_SubCat['Profit'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('Sub-Category')
plt.title('Top5 Sub-Category wise Profit')
plt.ylabel('Profit')


# In[223]:


df_SubCat=df_SubCat.groupby(['Sub-Category']).sum()
print(df_SubCat)


# In[224]:


df_SubCat['Profitability_Ratio']=df_SubCat['Profit']/df_SubCat['Sales']*100
df_SubCat


# In[225]:


df_SubCat['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='Sub-Category wise Profitability_Ratio',figsize=(15,7))


# In[226]:


df_SubCat.sort_values(by='Profitability_Ratio',ascending=False,inplace=True)
df_SubCat


# In[227]:


df_SubCat['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='Sub-Category wise Profitability_Ratio',figsize=(15,7))


# In[228]:


import matplotlib.pyplot as plt
top_5=df_SubCat['Profitability_Ratio'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('Sub-Category')
plt.title('Top5 Sub-Category wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[229]:


df_SubCat.sort_values(by='Profitability_Ratio',ascending=True,inplace=True)
df_SubCat


# In[230]:


df_SubCat['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='Sub-Category wise Profitability_Ratio',figsize=(15,7))


# In[231]:


import matplotlib.pyplot as plt
bottom_5=df_SubCat['Profitability_Ratio'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('Sub-Category')
plt.title('Bottom5 Sub-Category wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[232]:


import pandas as pd 
df_SM=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_SM


# In[233]:


df_SM=df_SM.groupby(['Ship Mode']).sum()
print(df_SM)


# In[234]:


df_SM['Sales'].plot(kind='bar',ylabel='Sales',title='Ship Mode wise Sales',figsize=(15,7))


# In[235]:


df_SM['Sales'].plot(kind='pie',ylabel='Sales',title='Ship Mode wise Sales',figsize=(15,7))


# In[236]:


df_SM['Profit'].plot(kind='bar',ylabel='Profit',title='Ship Mode wise Profit',figsize=(15,7))


# In[237]:


df_SM['Profit'].plot(kind='pie',ylabel='Profit',title='Ship Mode wise Profit',figsize=(15,7))


# In[238]:


df_SM['Profitabililty_Ratio']=df_SM['Profit']/df_SM['Sales']*100
df_SM


# In[239]:


df_SM['Profitabililty_Ratio'].plot(kind='bar',ylabel='Profitabililty_Ratio',title='Ship Mode wise Profitabililty_Ratio',figsize=(15,7))


# In[240]:


df_SM['Profitabililty_Ratio'].plot(kind='pie',ylabel='Profitabililty_Ratio',title='Ship Mode wise Profitabililty_Ratio',figsize=(15,7))


# In[241]:


import pandas as pd 
df_RE=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_RE


# In[242]:



df_RE=df_RE.query("Region == 'West'").groupby(["State"]).sum()
df_RE


# In[243]:


df_RE['Sales'].plot(kind='bar',ylabel='State wise Sales',title='West Region State wise Sales',figsize=(15,7))


# In[244]:


df_RE['Sales'].plot(kind='pie',ylabel='State wise Sales',title='West Region State wise Sales',figsize=(15,7))


# In[245]:


df_RE.sort_values(by='Sales',ascending=False,inplace=True)
df_RE


# In[246]:


df_RE['Sales'].plot(kind='bar',ylabel='State wise Sales',title='West Region State wise Sales',figsize=(15,7))


# In[248]:


df_RE['Sales'].plot(kind='pie',ylabel='State wise Sales',title='West Region State wise Sales',figsize=(15,7))


# In[249]:


import matplotlib.pyplot as plt
top_5=df_RE['Sales'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('West Region State wise Sales')
plt.title('Top5 West Region State Wise Sales')
plt.ylabel('Sales')


# In[250]:


df_RE.sort_values(by='Sales',ascending=True,inplace=True)
df_RE


# In[251]:


df_RE['Sales'].plot(kind='bar',ylabel='State wise Sales',title='West Region State wise Sales',figsize=(15,7))


# In[252]:


df_RE['Sales'].plot(kind='pie',ylabel='State wise Sales',title='West Region State wise Sales',figsize=(15,7))


# In[253]:


import matplotlib.pyplot as plt
bottom_5=df_RE['Sales'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('West Region State wise Sales')
plt.title('Bottom5 West Region State Wise Sales')
plt.ylabel('Sales')


# In[254]:


df_RE.sort_values(by='Profit',ascending=False,inplace=True)
df_RE


# In[255]:


df_RE['Profit'].plot(kind='bar',ylabel='State wise Profit',title='West Region State wise Profit',figsize=(15,7))


# In[256]:


import matplotlib.pyplot as plt
top_5=df_RE['Profit'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('West Region State wise Profit')
plt.title('Top5 West Region State Wise Profit')
plt.ylabel('Profit')


# In[257]:


df_RE.sort_values(by='Profit',ascending=True,inplace=True)
df_RE


# In[258]:


df_RE['Profit'].plot(kind='bar',ylabel='State wise Profit',title='West Region State wise Profit',figsize=(15,7))


# In[259]:


import matplotlib.pyplot as plt
bottom_5=df_RE['Profit'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('West Region State wise Profit')
plt.title('Bottom5 West Region State Wise Profit')
plt.ylabel('Profit')


# In[260]:



import pandas as pd 
df_RE=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_RE


# In[261]:



df_RE=df_RE.query("Region == 'West'").groupby(["State"]).sum()
df_RE


# In[262]:


df_RE['Profitability_Ratio']=df_RE['Profit']/df_RE['Sales']*100
df_RE


# In[263]:


df_RE['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='West Region State Wise Profitability_Ratio',figsize=(15,7))


# In[264]:


df_RE.sort_values(by='Profitability_Ratio',ascending=False,inplace=True)
df_RE


# In[265]:


df_RE['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='West Region State Wise Profitability_Ratio',figsize=(15,7))


# In[266]:


import matplotlib.pyplot as plt
top_5=df_RE['Profitability_Ratio'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('West Region State wise Profitability_Ratio')
plt.title('Top5 West Region State Wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[267]:


df_RE.sort_values(by='Profitability_Ratio',ascending=True,inplace=True)
df_RE


# In[268]:


df_RE['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='West Region State Wise Profitability_Ratio',figsize=(15,7))


# In[269]:


import matplotlib.pyplot as plt
bottom_5=df_RE['Profitability_Ratio'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('West Region State wise Profitability_Ratio')
plt.title('Bottom5 West Region State Wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[270]:



import pandas as pd 
df_RC=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_RC


# In[271]:



df_RC=df_RC.query("Region == 'Central'").groupby(["State"]).sum()
df_RC


# In[272]:


df_RC['Sales'].plot(kind='bar',ylabel='State wise Sales',title='Central Region State wise Sales',figsize=(15,7))


# In[273]:


df_RC['Sales'].plot(kind='pie',ylabel='State wise Sales',title='Central Region State wise Sales',figsize=(15,7))


# In[274]:


df_RC.sort_values(by='Sales',ascending=False,inplace=True)
df_RC


# In[275]:


df_RC['Sales'].plot(kind='bar',ylabel='State wise Sales',title='Central Region State wise Sales',figsize=(15,7))


# In[276]:


df_RC['Sales'].plot(kind='pie',ylabel='State wise Sales',title='Central Region State wise Sales',figsize=(15,7))


# In[277]:


import matplotlib.pyplot as plt
top_5=df_RC['Sales'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('Central Region State wise Sales')
plt.title('Top5 Central Region State Wise Sales')
plt.ylabel('Sales')


# In[278]:


df_RC.sort_values(by='Sales',ascending=True,inplace=True)
df_RC


# In[279]:


df_RC['Sales'].plot(kind='bar',ylabel='State wise Sales',title='Central Region State wise Sales',figsize=(15,7))


# In[280]:


df_RC['Sales'].plot(kind='pie',ylabel='State wise Sales',title='Central Region State wise Sales',figsize=(15,7))


# In[281]:


import matplotlib.pyplot as plt
bottom_5=df_RC['Sales'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('Central Region State wise Sales')
plt.title('Bottom5 - Central Region State Wise Sales')
plt.ylabel('Sales')


# In[282]:


df_RC.sort_values(by='Profit',ascending=False,inplace=True)
df_RC


# In[283]:


df_RC['Profit'].plot(kind='bar',ylabel='State wise Profit',title='Central Region State wise Profit',figsize=(15,7))


# In[284]:


import matplotlib.pyplot as plt
top_5=df_RC['Profit'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('Central Region State wise Profit')
plt.title('Top5 Central Region State Wise Profit')
plt.ylabel('Profit')


# In[285]:


df_RC.sort_values(by='Profit',ascending=True,inplace=True)
df_RC


# In[286]:


df_RC['Profit'].plot(kind='bar',ylabel='State wise Profit',title='Central Region State wise Profit',figsize=(15,7))


# In[287]:


import matplotlib.pyplot as plt
bottom_5=df_RC['Profit'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('Central Region State wise Profit')
plt.title('bottom5 - Central Region State Wise Profit')
plt.ylabel('Profit')


# In[288]:



import pandas as pd 
df_RC=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_RC


# In[289]:



df_RC=df_RC.query("Region == 'Central'").groupby(["State"]).sum()
df_RC


# In[290]:


df_RC['Profitability_Ratio']=df_RC['Profit']/df_RC['Sales']*100
df_RC


# In[291]:


df_RC['Profitability_Ratio'].plot(kind='bar',ylabel='State wise Profitability_Ratio',title='Central Region State wise Profitability_Ratio',figsize=(15,7))


# In[292]:


df_RC.sort_values(by='Profitability_Ratio',ascending=False,inplace=True)
df_RC


# In[293]:


import matplotlib.pyplot as plt
top_5=df_RC['Profitability_Ratio'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('Central Region State wise Profitability_Ratio')
plt.title('top5 - Central Region State Wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[294]:


df_RC.sort_values(by='Profitability_Ratio',ascending=True,inplace=True)
df_RC


# In[295]:


df_RC['Profitability_Ratio'].plot(kind='bar',ylabel='State wise Profitability_Ratio',title='Central Region State wise Profitability_Ratio',figsize=(15,7))


# In[296]:


import matplotlib.pyplot as plt
bottom_5=df_RC['Profitability_Ratio'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('Central Region State wise Profitability_Ratio')
plt.title('bottom5 - Central Region State Wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[297]:


import pandas as pd 
df_RS=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_RS


# In[298]:



df_RS=df_RS.query("Region == 'South'").groupby(["State"]).sum()
df_RS


# In[299]:


df_RS['Sales'].plot(kind='bar',ylabel='Sales',title='South Region wise Sales',figsize=(15,7))


# In[300]:


df_RS['Sales'].plot(kind='pie',ylabel='Sales',title='South Region wise Sales',figsize=(15,7))


# In[301]:


df_RS.sort_values(by='Sales',ascending=False,inplace=True)
df_RS


# In[302]:


df_RS['Sales'].plot(kind='bar',ylabel='Sales',title='South Region wise Sales',figsize=(15,7))


# In[304]:


df_RS['Sales'].plot(kind='pie',ylabel='Sales',title='South Region wise Sales',figsize=(15,7))


# In[305]:


import matplotlib.pyplot as plt
top_5=df_RS['Sales'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('South Region wise Sales')
plt.title('Top5 - South Region State wise Sales')
plt.ylabel('Sales')


# In[306]:


df_RS.sort_values(by='Sales',ascending=True,inplace=True)
df_RS


# In[307]:


df_RS['Sales'].plot(kind='bar',ylabel='Sales',title='South Region wise Sales',figsize=(15,7))


# In[308]:


df_RS['Sales'].plot(kind='pie',ylabel='Sales',title='South Region wise Sales',figsize=(15,7))


# In[309]:


import matplotlib.pyplot as plt
bottom_5=df_RS['Sales'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('South Region wise Sales')
plt.title('Bottom5 - South Region State wise Sales')
plt.ylabel('Sales')


# In[310]:


df_RS.sort_values(by='Profit',ascending=False,inplace=True)
df_RS


# In[311]:


df_RS['Profit'].plot(kind='bar',ylabel='Profit',title='South Region State wise Profit',figsize=(15,7))


# In[312]:


import matplotlib.pyplot as plt
top_5=df_RS['Profit'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('South Region wise Profit')
plt.title('Top5 - South Region State wise Profit')
plt.ylabel('Profit')


# In[313]:


df_RS.sort_values(by='Profit',ascending=True,inplace=True)
df_RS


# In[314]:


df_RS['Profit'].plot(kind='bar',ylabel='Profit',title='South Region State wise Profit',figsize=(15,7))


# In[315]:


import matplotlib.pyplot as plt
bottom_5=df_RS['Profit'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('South Region wise Profit')
plt.title('Bottom5 - South Region State wise Profit')
plt.ylabel('Profit')


# In[316]:


import pandas as pd 
df_RS=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_RS


# In[317]:



df_RS=df_RS.query("Region == 'South'").groupby(["State"]).sum()
df_RS


# In[318]:


df_RS['Profitability_Ratio']=df_RS['Profit']/df_RS['Sales']*100
df_RS


# In[319]:


df_RS['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='South Region State wise Profitability_Ratio',figsize=(15,7))


# In[320]:


df_RS.sort_values(by='Profitability_Ratio',ascending=False,inplace=True)
df_RS


# In[321]:


df_RS['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='South Region State wise Profitability_Ratio',figsize=(15,7))


# In[322]:


import matplotlib.pyplot as plt
top_5=df_RS['Profitability_Ratio'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('South Region wise Profitability_Ratio')
plt.title('Top5 - South Region State wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[323]:


df_RS.sort_values(by='Profitability_Ratio',ascending=True,inplace=True)
df_RS


# In[324]:


df_RS['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='South Region State wise Profitability_Ratio',figsize=(15,7))


# In[325]:


import matplotlib.pyplot as plt
bottom_5=df_RS['Profitability_Ratio'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('South Region wise Profitability_Ratio')
plt.title('Bottom5 - South Region State wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[326]:


import pandas as pd 
df_REst=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_REst


# In[327]:



df_REst=df_REst.query("Region == 'East'").groupby(["State"]).sum()
df_REst


# In[328]:


df_REst['Sales'].plot(kind='bar',ylabel='Sales',title='East Region State Wise Sales',figsize=(15,7))


# In[329]:


df_REst['Sales'].plot(kind='pie',ylabel='Sales',title='East Region State Wise Sales',figsize=(15,7))


# In[330]:


df_REst.sort_values(by='Sales',ascending=False,inplace=True)
df_REst


# In[331]:


df_REst['Sales'].plot(kind='bar',ylabel='Sales',title='East Region State Wise Sales',figsize=(15,7))


# In[332]:


df_REst['Sales'].plot(kind='pie',ylabel='Sales',title='East Region State Wise Sales',figsize=(15,7))


# In[333]:


import matplotlib.pyplot as plt
top_5=df_REst['Sales'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('East Region wise Sales')
plt.title('Top5 - East Region State wise Sales')
plt.ylabel('Sales')


# In[334]:


df_REst.sort_values(by='Sales',ascending=True,inplace=True)
df_REst


# In[335]:


df_REst['Sales'].plot(kind='bar',ylabel='Sales',title='East Region State Wise Sales',figsize=(15,7))


# In[336]:


df_REst['Sales'].plot(kind='pie',ylabel='Sales',title='East Region State Wise Sales',figsize=(15,7))


# In[337]:


import matplotlib.pyplot as plt
bottom_5=df_REst['Sales'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('East Region wise Sales')
plt.title('Bottom5 - East Region State wise Sales')
plt.ylabel('Sales')


# In[338]:


df_REst.sort_values(by='Profit',ascending=False,inplace=True)
df_REst


# In[339]:


df_REst['Profit'].plot(kind='bar',ylabel='Profit',title='East Region State Wise Profit',figsize=(15,7))


# In[340]:


import matplotlib.pyplot as plt
top_5=df_REst['Profit'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('East Region wise Profit')
plt.title('Top5 - East Region State wise Profit')
plt.ylabel('Profit')


# In[341]:


df_REst.sort_values(by='Profit',ascending=True,inplace=True)
df_REst


# In[342]:


df_REst['Profit'].plot(kind='bar',ylabel='Profit',title='East Region State Wise Profit',figsize=(15,7))


# In[343]:


import matplotlib.pyplot as plt
bottom_5=df_REst['Profit'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('East Region wise Profit')
plt.title('Bottom5 - East Region State wise Profit')
plt.ylabel('Profit')


# In[344]:


import pandas as pd 
df_REst=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_REst


# In[345]:



df_REst=df_REst.query("Region == 'East'").groupby(["State"]).sum()
df_REst


# In[346]:


df_REst['Profitability_Ratio']=df_REst['Profit']/df_REst['Sales']*100
df_REst


# In[347]:


df_REst['Profitability_Ratio'].plot(kind='bar',ylabel='	Profitability_Ratio',title='East Region State wise Profitability_Ratio',figsize=(15,7))


# In[348]:


df_REst.sort_values(by='Profitability_Ratio',ascending=False,inplace=True)
df_REst


# In[349]:


df_REst['Profitability_Ratio'].plot(kind='bar',ylabel='	Profitability_Ratio',title='East Region State wise Profitability_Ratio',figsize=(15,7))


# In[350]:


import matplotlib.pyplot as plt
top_5=df_REst['Profitability_Ratio'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('East Region wise Profitability_Ratio')
plt.title('Top5 - East Region State wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[351]:


df_REst.sort_values(by='Profitability_Ratio',ascending=True,inplace=True)
df_REst


# In[352]:


df_REst['Profitability_Ratio'].plot(kind='bar',ylabel='	Profitability_Ratio',title='East Region State wise Profitability_Ratio',figsize=(15,7))


# In[353]:


import matplotlib.pyplot as plt
bottom_5=df_REst['Profitability_Ratio'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('East Region wise Profitability_Ratio')
plt.title('Bottom5 - East Region State wise Profitability_Ratio')
plt.ylabel('Profitability_Ratio')


# In[354]:


import pandas as pd 
df_CS=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_CS


# In[355]:



df_CS=df_CS.query("State == 'California'").groupby(["City"]).sum()
df_CS


# In[356]:


df_CS['Sales'].plot(kind='bar',ylabel='Sales',title='City wise Sales from California State',figsize=(15,7))


# In[357]:


df_CS['Sales'].plot(kind='pie',ylabel='Sales',title='City wise Sales from California State',figsize=(15,7))


# In[358]:


df_CS.sort_values(by='Sales',ascending=False,inplace=True)
df_CS


# In[359]:


df_CS['Sales'].plot(kind='bar',ylabel='Sales',title='City wise Sales from California State',figsize=(15,7))


# In[360]:


df_CS['Sales'].plot(kind='pie',ylabel='Sales',title='City wise Sales from California State',figsize=(15,7))


# In[361]:


import matplotlib.pyplot as plt
top_5=df_CS['Sales'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('City Wise Sales from California State')
plt.title('Top5 -City Wise Sales from California State')
plt.ylabel('Sales')


# In[362]:


df_CS.sort_values(by='Sales',ascending=True,inplace=True)
df_CS


# In[363]:


df_CS['Sales'].plot(kind='bar',ylabel='Sales',title='City wise Sales from California State',figsize=(15,7))


# In[364]:


df_CS['Sales'].plot(kind='pie',ylabel='Sales',title='City wise Sales from California State',figsize=(15,7))


# In[366]:


import matplotlib.pyplot as plt
bottom_5=df_CS['Sales'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('City Wise Sales from California State')
plt.title('Bottom5 -City Wise Sales from California State')
plt.ylabel('Sales')


# In[367]:


df_CS.sort_values(by='Profit',ascending=False,inplace=True)
df_CS


# In[368]:


df_CS['Profit'].plot(kind='bar',ylabel='Profit',title='City wise Profit from California State',figsize=(15,7))


# In[369]:


import matplotlib.pyplot as plt
top_5=df_CS['Profit'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('City Wise Profit from California State')
plt.title('Top5 -City Wise Profit from California State')
plt.ylabel('Profit')


# In[370]:


df_CS.sort_values(by='Profit',ascending=True,inplace=True)
df_CS


# In[371]:


df_CS['Profit'].plot(kind='bar',ylabel='Profit',title='City wise Profit from California State',figsize=(15,7))


# In[372]:


import matplotlib.pyplot as plt
bottom_5=df_CS['Profit'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('City Wise Profit from California State')
plt.title('Bottom5 -City Wise Profit from California State')
plt.ylabel('Profit')


# In[373]:


import pandas as pd 
df_CS=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_CS


# In[374]:


df_CS=df_CS.query("State == 'California'").groupby(["City"]).sum()
df_CS


# In[375]:


df_CS['Profitability_Ratio']=df_CS['Profit']/df_CS['Sales']*100
df_CS


# In[376]:


df_CS['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='City wise Profitability_Ratio from California State',figsize=(15,7))


# In[377]:


df_CS.sort_values(by='Profitability_Ratio',ascending=False,inplace=True)
df_CS


# In[378]:


df_CS['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='City wise Profitability_Ratio from California State',figsize=(15,7))


# In[379]:


import matplotlib.pyplot as plt
top_5=df_CS['Profitability_Ratio'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('City wise Profitability_Ratio from California State')
plt.title('Top5 - City wise Profitability_Ratio from California State')
plt.ylabel('Profitability_Ratio')


# In[380]:


df_CS.sort_values(by='Profitability_Ratio',ascending=True,inplace=True)
df_CS


# In[381]:


df_CS['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='City wise Profitability_Ratio from California State',figsize=(15,7))


# In[382]:


import matplotlib.pyplot as plt
bottom_5=df_CS['Profitability_Ratio'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('City wise Profitability_Ratio from California State')
plt.title('Bottom5 - City wise Profitability_Ratio from California State')
plt.ylabel('Profitability_Ratio')


# In[383]:


import pandas as pd 
df_SC=pd.read_csv(r'C:\Raw Data\SampleSuperstore.csv',sep=',')
df_SC


# In[384]:


df_SC=df_SC.query("Category == 'Technology'").groupby(["State"]).sum()
df_SC


# In[385]:


df_SC['Sales'].plot(kind='bar',ylabel='Sales',title='State wise Sales from Category',figsize=(15,7))


# In[386]:


df_SC['Sales'].plot(kind='pie',ylabel='Sales',title='State wise Sales from Category',figsize=(15,7))


# In[387]:


df_SC.sort_values(by='Sales',ascending=False,inplace=True)
df_SC


# In[388]:


df_SC['Sales'].plot(kind='bar',ylabel='Sales',title='State wise Sales from Category',figsize=(15,7))


# In[389]:


df_SC['Sales'].plot(kind='pie',ylabel='Sales',title='State wise Sales from Category',figsize=(15,7))


# In[390]:


import matplotlib.pyplot as plt
top_5=df_SC['Sales'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Sales from Technology')
plt.title('Top5 - State wise Sales from Technology')
plt.ylabel('Sales')


# In[391]:


df_SC.sort_values(by='Sales',ascending=True,inplace=True)
df_SC


# In[392]:


df_SC['Sales'].plot(kind='bar',ylabel='Sales',title='State wise Sales from Technology',figsize=(15,7))


# In[393]:


df_SC['Sales'].plot(kind='pie',ylabel='Sales',title='State wise Sales from Technology',figsize=(15,7))


# In[394]:


import matplotlib.pyplot as plt
bottom_5=df_SC['Sales'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Sales of Technology')
plt.title('Bottom5 - State wise Sales of Technology')
plt.ylabel('Sales')


# In[395]:


df_SC.sort_values(by='Profit',ascending=False,inplace=True)
df_SC


# In[396]:


df_SC['Profit'].plot(kind='bar',ylabel='Profit',title='State wise Profit of Technology',figsize=(15,7))


# In[397]:


import matplotlib.pyplot as plt
top_5=df_SC['Profit'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Profit of Technology')
plt.title('Top5 - State wise Profit of Technology')
plt.ylabel('Profit')


# In[398]:


df_SC.sort_values(by='Profit',ascending=True,inplace=True)
df_SC


# In[399]:


df_SC['Profit'].plot(kind='bar',ylabel='Profit',title='State wise Profit of Technology',figsize=(15,7))


# In[400]:


import matplotlib.pyplot as plt
bottom_5=df_SC['Profit'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Profit of Technology')
plt.title('Bottom5 - State wise Profit of Technology')
plt.ylabel('Profit')


# In[401]:


import pandas as pd 
df_SR=pd.read_csv(r"C:\Raw Data\Sample Store.csv",sep=',')
df_SR


# In[402]:


df_SR=df_SR.query("Sub_Category == 'Phones'").groupby(["State"]).sum()
df_SR


# In[403]:


df_SR['Sales'].plot(kind='bar',ylabel='Sales',title='State wise Sales of Phones',figsize=(15,7))


# In[404]:


df_SR['Sales'].plot(kind='pie',ylabel='Sales',title='State wise Sales of Phones',figsize=(15,7))


# In[405]:


df_SR.sort_values(by='Sales',ascending=False,inplace=True)
df_SR


# In[406]:


df_SR['Sales'].plot(kind='bar',ylabel='Sales',title='State wise Sales of Phones',figsize=(15,7))


# In[407]:


df_SR['Sales'].plot(kind='pie',ylabel='Sales',title='State wise Sales of Phones',figsize=(15,7))


# In[369]:


import matplotlib.pyplot as plt
top_5=df_SR['Sales'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Sales of Phones')
plt.title('Top5 - State wise Sales of Phones')
plt.ylabel('Sales')


# In[370]:


df_SR.sort_values(by='Sales',ascending=True,inplace=True)
df_SR


# In[371]:


df_SR['Sales'].plot(kind='bar',ylabel='Sales',title='State wise Sales of Phones',figsize=(15,7))


# In[408]:


df_SR['Sales'].plot(kind='pie',ylabel='Sales',title='State wise Sales of Phones',figsize=(15,7))


# In[409]:


import matplotlib.pyplot as plt
bottom_5=df_SR['Sales'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Sales of Phones')
plt.title('Bottom5 - State wise Sales of Phones')
plt.ylabel('Sales')


# In[410]:


df_SR.sort_values(by='Profit',ascending=False,inplace=True)
df_SR


# In[411]:


df_SR['Profit'].plot(kind='bar',ylabel='Profit',title='State wise Profit of Phones',figsize=(15,7))


# In[412]:


import matplotlib.pyplot as plt
top_5=df_SR['Profit'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Profit of Phones')
plt.title('Top5 - State wise Profit of Phones')
plt.ylabel('Profit')


# In[413]:


df_SR.sort_values(by='Profit',ascending=True,inplace=True)
df_SR


# In[414]:


df_SR['Profit'].plot(kind='bar',ylabel='Profit',title='State wise Profit of Phones',figsize=(15,7))


# In[415]:


import matplotlib.pyplot as plt
bottom_5=df_SR['Profit'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Profit of Phones')
plt.title('Bottom5 - State wise Profit of Phones')
plt.ylabel('Profit')


# In[416]:


import pandas as pd 
df_SR=pd.read_csv(r"C:\Raw Data\Sample Store.csv",sep=',')
df_SR


# In[417]:


df_SR=df_SR.query("Sub_Category == 'Phones'").groupby(["State"]).sum()
df_SR


# In[418]:


df_SR['Profitability_Ratio']=df_SR['Profit']/df_SR['Sales']*100
df_SR


# In[419]:


df_SR['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='State wise Profitability_Ratio of Phones',figsize=(15,7))


# In[420]:


df_SR.sort_values(by='Profitability_Ratio',ascending=False,inplace=True)
df_SR


# In[421]:


df_SR['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='State wise Profitability_Ratio of Phones',figsize=(15,7))


# In[422]:


import matplotlib.pyplot as plt
top_5=df_SR['Profitability_Ratio'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Profitability_Ratio of Phones')
plt.title('Top5 - State wise Profitability_Ratio of Phones')
plt.ylabel('Profitability_Ratio')


# In[423]:


df_SR.sort_values(by='Profitability_Ratio',ascending=True,inplace=True)
df_SR


# In[424]:


df_SR['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='State wise Profitability_Ratio of Phones',figsize=(15,7))


# In[390]:


import matplotlib.pyplot as plt
bottom_5=df_SR['Profitability_Ratio'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Profitability_Ratio of Phones')
plt.title('Bottom5 - State wise Profitability_Ratio of Phones')
plt.ylabel('Profitability_Ratio')


# In[23]:


import pandas as pd 
df_CP=pd.read_csv(r"C:\Raw Data\Sample Store.csv",sep=',')
df_CP


# In[24]:


df_CP=df_CP.query("Sub_Category == 'Phones'").groupby(["City"]).sum()
df_CP


# In[25]:


df_CP['Sales'].plot(kind='bar',ylabel='Sales',title='City wise Sales of Phones',figsize=(15,7))


# In[26]:


df_CP['Sales'].plot(kind='pie',ylabel='Sales',title='City wise Sales of Phones',figsize=(15,7))


# In[27]:


df_CP.sort_values(by='Sales',ascending=False,inplace=True)
df_CP


# In[28]:


df_CP['Sales'].plot(kind='bar',ylabel='Sales',title='City wise Sales of Phones',figsize=(15,7))


# In[29]:


df_CP['Sales'].plot(kind='pie',ylabel='Sales',title='City wise Sales of Phones',figsize=(15,7))


# In[425]:


import matplotlib.pyplot as plt
top_5=df_CP['Sales'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('City wise Sales of Phones')
plt.title('Top5 - City wise Sales of Phones')
plt.ylabel('Sales')


# In[426]:


df_CP.sort_values(by='Sales',ascending=True,inplace=True)
df_CP


# In[427]:


df_CP['Sales'].plot(kind='bar',ylabel='Sales',title='City wise Sales of Phones',figsize=(15,7))


# In[428]:


df_CP['Sales'].plot(kind='pie',ylabel='Sales',title='City wise Sales of Phones',figsize=(15,7))


# In[429]:


import matplotlib.pyplot as plt
bottom_5=df_CP['Sales'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('City wise Sales of Phones')
plt.title('Bottom5 - City wise Sales of Phones')
plt.ylabel('Sales')


# In[430]:


df_CP.sort_values(by='Profit',ascending=False,inplace=True)
df_CP


# In[431]:


df_CP['Profit'].plot(kind='bar',ylabel='Profit',title='City wise Profit of Phones',figsize=(15,7))


# In[432]:


import matplotlib.pyplot as plt
top_5=df_CP['Profit'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('City wise Profit of Phones')
plt.title('Top5 - City wise Profit of Phone')
plt.ylabel('Profit')


# In[433]:


df_CP.sort_values(by='Profit',ascending=True,inplace=True)
df_CP


# In[434]:


df_CP['Profit'].plot(kind='bar',ylabel='Profit',title='City wise Profit of Phones',figsize=(15,7))


# In[435]:


import matplotlib.pyplot as plt
bottom_5=df_CP['Profit'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('City wise Profit of Phones')
plt.title('Bottom5 - City wise Profit of Phone')
plt.ylabel('Profit')


# In[436]:


import pandas as pd 
df_CP=pd.read_csv(r"C:\Raw Data\Sample Store.csv",sep=',')
df_CP


# In[437]:


df_CP=df_CP.query("Sub_Category == 'Phones'").groupby(["City"]).sum()
df_CP


# In[438]:


df_CP['Profitability_Ratio']=df_CP['Profit']/df_CP['Sales']*100
df_CP


# In[439]:


df_CP['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='City wise Profitability_Ratio of Phones',figsize=(15,7))


# In[440]:


df_CP.sort_values(by='Profitability_Ratio',ascending=False,inplace=True)
df_CP


# In[441]:


df_CP['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='City wise Profitability_Ratio of Phones',figsize=(15,7))


# In[442]:


import matplotlib.pyplot as plt
top_5=df_CP['Profitability_Ratio'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('City wise Profitability_Ratio of Phones')
plt.title('Top5 - City wise Profitability_Ratio of Phones')
plt.ylabel('Profitability_Ratio')


# In[443]:


df_CP.sort_values(by='Profitability_Ratio',ascending=True,inplace=True)
df_CP


# In[444]:


df_CP['Profitability_Ratio'].plot(kind='bar',ylabel='Profitability_Ratio',title='City wise Profitability_Ratio of Phones',figsize=(15,7))


# In[445]:


import matplotlib.pyplot as plt
bottom_5=df_CP['Profitability_Ratio'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('City wise Profitability_Ratio of Phones')
plt.title('Bottom5 - City wise Profitability_Ratio of Phones')
plt.ylabel('Profitability_Ratio')


# In[446]:


import pandas as pd 
df_Q=pd.read_csv(r"C:\Raw Data\Sample Store.csv",sep=',')
df_Q


# In[447]:


df_Q=df_Q.groupby(['Sub_Category','State']).sum()
print(df_Q)


# In[448]:


df_Q['Profit'].plot(kind='bar',ylabel='Profit',title='State wise Profit of Sub_Category',figsize=(15,7))


# In[449]:


df_Q.sort_values(by='Profit',ascending=False,inplace=True)
df_Q


# In[450]:


df_Q['Profit'].plot(kind='bar',ylabel='Profit',title='State wise Profit of Sub_Category',figsize=(15,7))


# In[451]:


import matplotlib.pyplot as plt
top_5=df_Q['Profit'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Profit of Sub_Category')
plt.title('Top5 - State wise Profit of Sub_Category')
plt.ylabel('Profit')


# In[452]:


df_Q.sort_values(by='Profit',ascending=True,inplace=True)
df_Q


# In[453]:


df_Q['Profit'].plot(kind='bar',ylabel='Profit',title='State wise Profit of Sub_Category',figsize=(15,7))


# In[454]:


import matplotlib.pyplot as plt
bottom_5=df_Q['Profit'].iloc[:5]
bottom_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Profit of Sub_Category')
plt.title('Bottom5 - State wise Profit of Sub_Category')
plt.ylabel('Profit')


# In[455]:


df_Q=df_Q.query("Sub_Category == 'Phones'").groupby(["State"]).sum()
df_Q


# In[456]:


df_Q['Profit_Per_Piece']=df_Q['Profit']/df_Q['Quantity']
df_Q['Sales_Per_Piece']=df_Q['Sales']/df_Q['Quantity']
df_Q['Buy_Price_Per_Piece']=df_Q['Sales_Per_Piece']-df_Q['Profit_Per_Piece']
df_Q


# In[457]:


df_Q.sort_values(by='Profit_Per_Piece',ascending=False,inplace=True)
df_Q


# In[458]:


df_Q['Profit_Per_Piece'].plot(kind='bar',ylabel='Profit_Per_Piece',title='State wise Profit_Per_Piece of Phone',figsize=(15,7))


# In[459]:


import matplotlib.pyplot as plt
top_5=df_Q['Profit_Per_Piece'].iloc[:5]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Profit_Per_Piece of Phone')
plt.title('Top5 - State wise Profit_Per_Piece of Phone')
plt.ylabel('Profit_Per_Piece')


# In[460]:


df_Q.sort_values(by='Quantity',ascending=False,inplace=True)
df_Q


# In[461]:


df_Q['Quantity'].plot(kind='bar',ylabel='Quantity',title='State wise Quantity of Phones',figsize=(15,7))


# In[462]:


import matplotlib.pyplot as plt
top_5=df_Q['Quantity'].iloc[:10]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Quantity of Phones')
plt.title('Top10 - State wise Quantity of Phones')
plt.ylabel('Quantity')
plt.xticks(rotation='vertical')


# In[463]:


import matplotlib.pyplot as plt
top_5=df_Q['Profit_Per_Piece'].iloc[:10]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Profit_Per_Piece of Phones')
plt.title('Top10 - State wise Profit_Per_Piece of Phones')
plt.ylabel('Profit_Per_Piece')
plt.xticks(rotation='vertical')


# In[464]:


import matplotlib.pyplot as plt
top_5=df_Q['Sales_Per_Piece'].iloc[:10]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Sales_Per_Piece of Phones')
plt.title('Top10 - State wise Sales_Per_Piece of Phones')
plt.ylabel('Sales_Per_Piece')
plt.xticks(rotation='vertical')


# In[465]:


import matplotlib.pyplot as plt
top_5=df_Q['Buy_Price_Per_Piece'].iloc[:10]
top_5.plot(kind='bar',edgecolor='black',rot=0)
plt.xlabel('State wise Buy_Price_Per_Piece of Phones')
plt.title('Top10 - State wise Buy_Price_Per_Piece of Phones')
plt.ylabel('Buy_Price_Per_Piece')
plt.xticks(rotation='vertical')


# In[466]:


df_Q.sort_values(by='Profit_Per_Piece',ascending=False,inplace=True)
df_Q


# In[467]:


print(df_Q.head(5))


# In[468]:


print(df_Q.iloc[:3])


# In[55]:


df_T=(df_Q[['Buy_Price_Per_Piece','Sales_Per_Piece','Profit_Per_Piece']].head(3))
df_T


# In[56]:


axes = df_T.plot.bar(rot=0, subplots=True)
axes[1].legend(loc=2)  


# In[57]:


ax = df_T.plot.bar(stacked=True)
ax


# In[58]:


axes = df_T.plot.bar(
    rot=0, subplots=True, color={"Buy_Price_Per_Piece": "red", "Sales_Per_Piece": "green","Profit_Per_Piece":'violet'}
)
axes[1].legend(loc=2) 


# In[59]:


import scipy.stats as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
df_hmp=pd.read_csv(r'C:\Raw Data\Sample Store.csv',sep=',')
np.corrcoef(df_hmp['Sales'],df_hmp['Profit'])
sns.heatmap(df_hmp.corr())


# In[ ]:





# In[ ]:




