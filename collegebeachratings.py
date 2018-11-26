import pandas as pd
from trueskill import Rating, quality_1vs1, rate_1vs1, quality, rate, global_env, choose_backend,  backends, setup
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
from IPython.display import display
from ipywidgets import widgets
display(HTML("<style>.container { width:90% !important; }</style>"))

sort_order = ['date', 'compID', 'position', 'result']

#UTILITIES
def strip_all_strings_in_df(df_, cols_):
    for s in cols_:
        df_[s] = df_[s].apply(lambda x: x.strip())
    return df_

#LOAD DATA
def get_competition_info(df_, div_='DI'):
    df_ = get_teams_by_division(df_, div_)
    df_ = strip_all_strings_in_df(df_, ['first_name', 'last_name', 'college', 'location', 'division'])
    df_['date'] = pd.to_datetime(df_['date'], errors = 'coerce')

    df_ = df_.groupby('compID') \
        .filter(lambda x: len(x) > 12) \ #this needed to filter out division 1 teams that play against other divisions
        .sort_values(['date', 'compID', 'position', 'result'], ascending=True) \
        .reset_index() \
        .iloc[:,1:]
    return df_

#GET TEAM INFO
def get_teams_by_division(df_, div_='DI'):
    df_ = df_[df_["division"] == div_].drop_duplicates() \
        .reset_index().iloc[:,1:] \
        .sort_values(sort_order, ascending=True)
    return df_

#SETTING RATINGS
def set_default_player_rating(df_, div_='DI',  dict_start_rating_ = None):
    df_players_ = df_.loc[:, ['division',  'college', 'first_name', 'last_name', 'mu', 'sigma']].copy() \
                    .drop_duplicates()
    df_players_ = strip_all_strings_in_df(df_players_, ['division',  'college', 'first_name', 'last_name'])
    df_players_ = df_players_[df_players_.division == div_]
    df_players_.loc[:,'mu'] = Rating().mu
    df_players_['sigma'] = Rating().sigma

    #
    if dict_start_rating_ != None:
        df_players_['name'] = df_players_['first_name'] + " " + df_players_['last_name']
        df_players_['rating'] = df_players_['name'].replace(dict_start_ratings)
        df_players_['mu'] = df_players_['rating'].where(df_players_['rating'].isin([25,26,27,28,29])).fillna(25)
    #

    df_players_ = df_players_.sort_values(['last_name', 'first_name', 'college'], ascending=True) \
        .drop_duplicates() \
        .reset_index() \
        .loc[:, ['division',  'college', 'first_name', 'last_name',  'mu', 'sigma']]

    return df_players_


def create_start_rating(_adder_dict, _df_top_players):
    df_adder= pd.DataFrame.from_dict(_adder_dict, orient='index', columns=['adder']).reset_index()
    df_adder.columns = ['name', 'adder']
    df_starting_rating = pd.merge(_df_top_players, df_adder, how='outer')
    df_starting_rating['adder'] = df_starting_rating['adder'].fillna(0)
    df_starting_rating['start_rating'] = df_starting_rating['adder']+25
    df_starting_rating.columns = ['rank', 'name', 'college', 'division', 'last_season_end_rating',
       'primary position', 'date', 'adder', 'new_season_start_rating']
    return df_starting_rating[['name', 'last_season_end_rating','new_season_start_rating']]


def create_ratings_from_matches(df_, k_factor_=[0,0,0,0,0], div_='DI', dict_start_rating_ = None):
    sort_order = ['date', 'compID', 'position', 'result']
    df_matches = get_competition_info(df_, div_)
    df_players = set_default_player_rating(df_, div_, dict_start_rating_)

    groups = df_matches.groupby(['date', 'compID', 'position']).groups
    counter = 0
    hist_dict = {}

    for key in groups.keys():
        value = groups[key]

        r = {}
        r_new = {}

        for i in range(len(value)):
            fname = df_matches.iloc[value[i], df_matches.columns.get_loc('first_name')]
            lname = df_matches.iloc[value[i], df_matches.columns.get_loc('last_name')]
            college = df_matches.iloc[value[i], df_matches.columns.get_loc('college')]
            position = df_matches.iloc[value[i], df_matches.columns.get_loc('position')]
            r[i] = get_player_ratings_from_match(df_players, fname, lname, college)

        t1 = [r[0], r[1]]
        t2 = [r[2], r[3]]

        ((r_new[0], r_new[1]), (r_new[2], r_new[3])) = rate([t1,t2], ranks=[1,0])

        for j in range(4):
            fname = df_matches.iloc[value[j], df_matches.columns.get_loc('first_name')]
            lname = df_matches.iloc[value[j], df_matches.columns.get_loc('last_name')]
            college = df_matches.iloc[value[j], df_matches.columns.get_loc('college')]
            position = df_matches.iloc[value[j], df_matches.columns.get_loc('position')]
            set_player_ratings_after_match(df_players, \
                                           fname, \
                                           lname, \
                                           college, \
                                           position, \
                                           k_factor_, \
                                           Rating(mu=(r_new[j].mu +k_factor_[i]), sigma=r_new[j].sigma  ))

        hist_dict[counter] = df_matches.iloc[value].merge(df_players, \
                                                          left_on=['first_name', 'last_name', 'college'],
                                                          right_on=['first_name', 'last_name', 'college'],
                                                          how='inner')
        counter += 1

    df_match_history = pd.concat(hist_dict.values(), axis=0) \
                        .sort_values(['date','compID'], ascending=True) \
                        .reset_index() \
                        .iloc[:,1:] \
                        .rename(columns={'division_y': 'division'})
    df_match_history['name'] = df_match_history['first_name'] + " " + df_match_history['last_name']


    df_match_history = set_primary_position( df_match_history)

    d = {'matches': df_matches, 'match_history': df_match_history, 'players':  df_players}
    return d


#GETTING RATINGS
def get_player_ratings_from_match(df_, fname_, lname_, college_):
    player_idx = df_[(df_.first_name == fname_) \
                 & (df_.last_name == lname_) \
                 & (df_.college == college_)].index
    my_mu = df_.iloc[player_idx.item(),  df_.columns.get_loc('mu')]
    my_sigma = df_.iloc[player_idx.item(),  df_.columns.get_loc('sigma')]
    my_rating = Rating(mu=my_mu, sigma=my_sigma)
    return my_rating


def set_player_ratings_after_match(df_, fname_, lname_, college_, position_,k_factor_, rating_):
    player_idx = df_[(df_.first_name == fname_) \
                 & (df_.last_name == lname_) \
                 & (df_.college == college_)].index
    df_.iloc[player_idx,  df_.columns.get_loc('mu')] = rating_.mu + k_factor_[int(position_)-1]
    #df_.iloc[player_idx,  df_.columns.get_loc('mu')] = rating_.mu
    df_.iloc[player_idx,  df_.columns.get_loc('sigma')] = rating_.sigma

    my_mu = df_.iloc[player_idx.item(),  df_.columns.get_loc('mu')]
    my_sigma = df_.iloc[player_idx.item(),  df_.columns.get_loc('sigma')]
    my_rating = Rating(mu=my_mu, sigma=my_sigma)
    return my_rating


# TRACK RATINGS
#Get player info
def set_primary_position(df_):
    df_['count_max'] = df_.groupby(['college', 'name', 'position'])['position'].transform('count')
    df_['count at primary position'] = df_.groupby(['college','name'])['count_max'].transform('max')
    idx = df_[df_['count at primary position'] == df_['count_max']]

    #primary_position = strip_all_strings_in_df(primary_position, ['college','name'])
    primary_position = idx.loc[:, ['college','name', 'position']]
    primary_position['position'] = primary_position.groupby(['name'])['position'].transform('min')
    primary_position = primary_position.drop_duplicates()

    temp = df_.merge(primary_position, on=['college', 'name'])
    temp.rename(columns = {'position_y': 'primary position', 'position_x': 'position'}, inplace=True)
    temp = temp.loc[:,['date','compID', 'location', 'first_name', 'last_name', 'name', \
          'position', 'primary position', 'college','division' ,'result', 'mu', 'sigma' ]]
    return temp

def get_top_players(df_, position_='all'):
    if position_ == 'all':
        df_top_players_ = df_.sort_values(['date', 'compID'], ascending=True).copy()
    else:
        df_top_players_ = df_[df_['primary position'] == position_].sort_values(['date', 'compID'], ascending=True).copy()
    df_top_players_ = df_top_players_.groupby(['first_name', 'last_name', 'college', 'division']) \
                                    .last() \
                                    .sort_values('mu', ascending=False) \
                                    .reset_index()

    df_top_players_['rank'] = df_top_players_.index+1
    df_top_players_ = df_top_players_.loc[:,['rank','name', 'college', 'division', 'mu', 'primary position', 'date']]
    df_top_players_.columns= ['rank','name', 'college', 'division', 'rating', 'primary position' ,'date']
    return df_top_players_

def get_player_rating(df, a):
    temp_df = df[df['name'] == a]
    r = Rating(temp_df.groupby(['mu'])['date'].last().sort_values(ascending=False).index[0], temp_df.groupby(['sigma'])['date'].last().sort_values(ascending=False).index[0])
    return r

def get_player_rating_history(df, a):
    temp_df = df[df['name'] == a]
    #print temp_df
    print(a.upper())
    print("College: " + temp_df['college'].unique()[-1])
    print("Division: " + temp_df['division'].unique()[-1])
    print("Primary Position: " + str(temp_df['primary position'].unique()[-1]))
    print("Last Rating:" + str(temp_df.groupby(['mu'])['date'].last().sort_values(ascending=False).index[0]))
    f =  plt.figure(figsize=(30, 15))
    sns.lineplot(x="date", y="mu", data=temp_df)

    temp_df['opponent_result'] = abs(temp_df.result - 1)

    df_partner = pd.merge(df[df['name'] != a],temp_df, left_on=['compID', 'position', 'result'], right_on=['compID', 'position', 'result'] ) \
                    .groupby(['compID']) \
                    .first()
    df_partner.reset_index().set_index('compID')
    df_partner['Partner'] = df_partner['name_x']+' '+df_partner['mu_x'].apply(lambda x: '('+str(x)+ ')')
    df_partner = df_partner[['Partner']]

    df_summary = pd.merge(temp_df,df_partner, on='compID')

    df_opp = pd.merge(df[['name', 'mu' ,'compID', 'position', 'result']], \
                       temp_df[['compID', 'position', 'opponent_result']],
                       left_on=['compID', 'position', 'result'],
                       right_on=['compID', 'position', 'opponent_result']) \
                .groupby(['compID'])

    df_opp1 = df_opp.first().reset_index()[['compID','name', 'mu']]
    df_opp1['Opponent 1'] = df_opp1['name']+' '+df_opp1['mu'].apply(lambda x: '('+str(x)+ ')')
    df_opp2 = df_opp.last().reset_index()[['compID','name', 'mu']]
    df_opp2['Opponent 2'] = df_opp2['name']+' '+df_opp2['mu'].apply(lambda x: '('+str(x)+ ')')

    opp = pd.concat([df_opp1, df_opp2], axis=1)
    opp.columns = ['compID', 'name', 'mu', 'Opponent 1', 'compID1', 'name', 'mu','Opponent 2']
    opp = opp[['compID', 'Opponent 1', 'Opponent 2']]

    df_summary = pd.merge(df_summary,opp, on='compID')
    df_summary = df_summary[['date', 'compID', 'location', 'first_name', 'last_name', 'name', \
       'position', 'primary position', 'college', 'division', 'result', \
       'mu', 'sigma', 'Partner', 'Opponent 1', 'Opponent 2']]

    return df_summary

def plot_team_players_history(df, college):
    df_team_history = df[df.college == college].sort_values('date', ascending=True)
    playercount = len(df_team_history.name.unique())
    palette = sns.color_palette("husl", playercount)
    f =  plt.figure(figsize=(30, 15))
    sns.lineplot(data=df_team_history, x="date", y="mu", hue="name", \
                 style="primary position", markers=True, palette=palette,ci=None)

def get_prior_year_player_rating_adders(df_top_players_):
    #2 standard deviations from the mean - best performers
    two_sigma = 2*df_top_players_.rating.std() + df_top_players_.rating.mean()
    high_perfomers = df_top_players_[df_top_players_['rating'] > two_sigma][['name']]
    high_perfomers['adder'] = 2

    #3 standard deviations from the mean - over-performing outliers
    three_sigma = 3*df_top_players_.rating.std() + df_top_players_.rating.mean()
    ultra_high_perfomers = df_top_players_[df_top_players_['rating'] > three_sigma][['name']]
    ultra_high_perfomers['adder'] = 1

    adder = pd.concat([high_perfomers, ultra_high_perfomers])
    adder = adder.groupby(['name'])['adder'].sum().to_frame().sort_values(['adder'], ascending=False)
    adder_dict =  adder.to_dict()['adder']
    return adder_dict

# Plotting
def plot_rating_distribution(df_, save_path_=None):
    d = df_ \
        .sort_values('date', ascending=True) \
        .groupby(['first_name', 'last_name', 'college']) \
        .last() \
        .sort_values('mu', ascending=False) \
        .reset_index()

    current_palette = sns.color_palette()

    f =  plt.figure(figsize=(30, 15))
    grid = plt.GridSpec(3, 5, wspace=.4, hspace=0.3)
    summary_ax = f.add_subplot(grid[:2, 0:])
    pos_ax = [f.add_subplot(grid[2, 4+(i*-1)], xticklabels=[], sharey=summary_ax) for i in range(5)]
    [sns.distplot( d[d['primary position'] == i+1]['mu'], ax=summary_ax,  color=current_palette[i], axlabel='Rating', hist_kws=dict(alpha=.1)) for i in range(5)]
    [sns.distplot( d[d['primary position'] == i+1]['mu'], ax=pos_ax[i], color=current_palette[i], axlabel='Position '+str(i+1))  for i in range(5)]

    if save_path_ != None:
        f.savefig(save_path_)

    return d

def plot_top_player_rating_distribution(df_, save_path_=None):
    f =  plt.figure(figsize=(20, 7))
    sns.distplot( df_['rating'], hist_kws=dict(alpha=.1))
    #sns.lineplot( df_['rating'])
    print(df_['rating'].mean(), df_['rating'].std())
    pyplot.axvline(x=df_['rating'].mean())
    pyplot.axvline(x=df_['rating'].mean()+ df_['rating'].std(), linestyle='-.')
    pyplot.axvline(x=df_['rating'].mean()- df_['rating'].std(), linestyle='-.')
    pyplot.axvline(x=df_['rating'].mean()+ 2*df_['rating'].std(), linestyle='--')
    pyplot.axvline(x=df_['rating'].mean()- 2*df_['rating'].std(), linestyle='--')
    pyplot.axvline(x=df_['rating'].mean()+ 3*df_['rating'].std(), linestyle=':')

    if save_path_ != None:
        f.savefig(save_path_)
    return None

## Get Starting Ratings from 2017
df_2017 = pd.read_csv('Data/Competition-2017.csv').drop_duplicates()
df_2017.columns = ['division', 'compID', 'location', 'date', 'college', 'first_name', 'last_name', 'position', 'result', 'score']
spread_factor_minimized = [1.1, .9, .5, .3, .05]
d_2017 = create_ratings_from_matches(df_2017, spread_factor_minimized)

plot_rating_distribution(d_2017['match_history'],'2017_110_090_050_030_005/rating_distribution.pdf')

df_top_players_2017 = get_top_players(d_2017['match_history'], position_='all')
df_top_players_2017.head(100).to_csv('2017_110_090_050_030_005/top_players.csv')


dict_adder_from_2017 = get_prior_year_player_rating_adders(df_top_players_2017)

df_start_rating_2018 = create_start_rating(dict_adder_from_2017, df_top_players_2017)
df_start_rating_2018.to_csv('2017_110_090_050_030_005/top_players_next_year_starting_rating.csv')


dict_start_ratings = df_start_rating_2018[['name','new_season_start_rating' ]].set_index('name').to_dict(orient='dict')['new_season_start_rating']
dict_start_ratings

## Process 2018
df_2018 = pd.read_csv('Data/Competition.csv').drop_duplicates()
df_2018.columns = ['division', 'compID', 'location', 'date', 'college', 'first_name', 'last_name', 'position', 'result', 'score']

d_2018 = create_ratings_from_matches(df_2018, spread_factor_minimized, 'DI', dict_start_ratings)

!mkdir 2018_110_090_050_030_005
plot_rating_distribution(d_2018['match_history'],'2018_110_090_050_030_005/rating_distribution.pdf')

df_top_players_2018 = get_top_players(d_2018['match_history'], position_='all')
df_top_players_2018.head(100).to_csv('2018_110_090_050_030_005/top_players.csv')
df_top_players_2018

#d_2017['matches'][d_2017['matches']['last_name'] == 'Sponcil']


get_player_rating_history(d_2018['match_history'], 'Sarah Sponcil')


dict_adder_2018 = get_prior_year_player_rating_adders(df_top_players_2018)
df_start_rating_2019 = create_start_rating(dict_adder_2018, df_top_players_2018)
df_start_rating_2019.to_csv('2018_110_090_050_030_005/top_players_next_year_starting_rating.csv')

dict_start_ratings = df_start_rating_2019[['name','new_season_start_rating' ]].set_index('name').to_dict(orient='dict')['new_season_start_rating']
dict_start_ratings


plot_team_players_history(d_2018['match_history'], 'USC')