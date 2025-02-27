from _util import *
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from matplotlib.transforms import BlendedGenericTransform
from matplotlib.offsetbox import AnchoredText
def get_tableau20():
    # These are the "Tableau 20" colors as RGB.   
    tableau20 = [(31, 119, 180), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)    
    return tableau20


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

class Analyzer():
    def __init__(self):
        pass

    def recover_full_recorder(self, record_R_only = None, only_plot_matrix = True):
        if only_plot_matrix:
            self.names = self.agent_names = record_R_only['name']
        else:
            self.names = self.agent_names = agent_names = list(record_R_only['record'][0].keys())
        self.record = {}
        self.setting = record_R_only['setting']
        self.record_R_only = record_R_only
        self.only_plot_matrix = only_plot_matrix
        if not only_plot_matrix:
            seeds = list(record_R_only['record'].keys())
            for seed in tqdm(seeds):
                self.record[seed] = self.recover_full_recorder_one_seed(seed)
                        
    def recover_full_recorder_one_seed(self, seed):
        r = {}
        for metric in ['R', 'A', 'regret', 'meta_regret']:
            r[metric] = {name : [] for name in self.agent_names}

        r['R'] = self.record_R_only['record'][seed]
        for name in self.agent_names:
            r['regret'][name] = arr(r['R']["oracle"]) - arr(r['R'][name])
        r['cum_regret'] = {name : np.cumsum(r['regret'][name]) for name in self.agent_names}
        r['cum_R'] = {name : np.cumsum(r['R'][name]) for name in self.agent_names}
        #r['cum_exp_R'] = {name : np.cumsum(r['exp_R'][name]) for name in self.agent_names}
        # x: time, y: cum_regret: group, name
        if "oracle-TS" in self.agent_names:
            for name in self.agent_names:
                r['meta_regret'][name] = arr(r['R']['oracle-TS']) - arr(r['R'][name])
            r['cum_meta_regret'] = {name : np.cumsum(r['meta_regret'][name]) for name in self.agent_names}
        return r
    
 


                
    def organize_Df(self, r_dict):
        T = len(r_dict[self.agent_names[0]])
        a = pd.DataFrame.from_dict(r_dict)
        a = pd.melt(a)
        a['time'] = np.tile(np.arange(T), len(self.agent_names))
        a = a.rename(columns = {'variable':'method'
                           , "value" : "regret"
                           , "time" : "time"})
        return a

    def prepare_data_4_plot(self, skip_methods = [], plot_mean = None, skip = None, plot_which = None):
        self.plot_which = plot_which
        # https://seaborn.pydata.org/generated/seaborn.lineplot.html
        #ax.legend(['label 1', 'label 2'])

        n_methods = 7# - len(skip_methods)

        reps = len(self.record)

        ########
        if self.only_plot_matrix:
            data_plot_BR = self.record_R_only['record']['data_plot_BR_original']
            if plot_which == 'R' or plot_which == "exp_R":
                data_plot_R = self.record_R_only['record']['data_plot_R_original']
        else:
            data = pd.concat([self.organize_Df(self.record[seed]['cum_regret']) for seed in range(reps)])
            data_R = pd.concat([self.organize_Df(self.record[seed]['cum_R']) for seed in range(reps)])

            if self.setting['order'] == "sequential":
                data_plot_BR = data.iloc[np.arange(0, len(data), step = self.setting['T']) + self.setting['T'] - 1]   
                data_plot_BR.time = np.tile(np.arange(0, self.setting['M'])
                                            , len(data_plot_BR) // self.setting['M'])
                data_plot_R = data_R.iloc[np.arange(0, len(data_R), step = self.setting['T']) + self.setting['T'] - 1]   
                data_plot_R.time = np.tile(np.arange(0, self.setting['M'])
                                            , len(data_plot_R) // self.setting['M'])
                

            elif self.setting['order'] == "concurrent":
                data_plot_BR = data.iloc[np.arange(0, len(data), step = self.setting['M']) + self.setting['M'] - 1]   
                data_plot_BR.time = np.tile(np.arange(0, self.setting['T'])
                                            , len(data_plot_BR) // self.setting['T'])
                data_plot_R = data_R.iloc[np.arange(0, len(data_R), step = self.setting['M']) + self.setting['M'] - 1]   
                data_plot_R.time = np.tile(np.arange(0, self.setting['T'])
                                            , len(data_plot_R) // self.setting['T'])
                
        self.data_plot_BR_original = data_plot_BR.copy()

        if plot_mean:
            data_plot_BR.regret = data_plot_BR.regret / (data_plot_BR.time + 1)

        data_plot_BR = data_plot_BR.rename(columns = {'method':'Method'})
        
        if plot_which == 'R' or plot_which == "exp_R":
            self.data_plot_R_original = data_plot_R.copy()
            if plot_mean:
                data_plot_R.regret = data_plot_R.regret / (data_plot_R.time + 1)
            data_plot_R = data_plot_R.rename(columns = {'method':'Method'})

            
        skip_methods.append("oracle")
        if skip_methods is not None:
            for met in skip_methods:
                data_plot_BR = data_plot_BR[data_plot_BR.Method != met]

        data_plot_BR = data_plot_BR[data_plot_BR.time >= skip]

        if plot_which == 'R' or plot_which == "exp_R":
            return data_plot_BR, data_plot_R
        
        return data_plot_BR

    def plot_regret(self, skip_methods = ["OSFA"]
                    , ci = None, freq = 20
                    , plot_which = "both", plot_mean = False, skip = 2
                   , n_boot = 50, ylabel = "Average regret"
                   , ax1 = None, w_title = False, y_min = None, y_max = None, i = 0, new_title = None
                   , color_shift = 0, palette_idx = None
                   , linewidth = 2, no_xtick = False
                    , hue_order = None
                    , complex_x_label = True,
                    w_x_label = True
                   ):
        from matplotlib.transforms import BlendedGenericTransform
        COLORS = get_tableau20() #sns.color_palette("tab10")
        if palette_idx is None:
            def rotate(l, n):
                n = -n
                return l[n:] + l[:n]
            palette = {name : color for name, color in zip(rotate(self.names, color_shift), COLORS)}
        else:
            palette = {name : COLORS[idx] for name, idx in palette_idx.items()}
           
        
        if plot_which == 'R' or plot_which == "exp_R":
            data_plot_BR, data_plot_R = self.prepare_data_4_plot(skip_methods = skip_methods
                                                                , plot_mean = plot_mean, skip = skip, plot_which = plot_which)

        else:
            data_plot_BR = self.prepare_data_4_plot(skip_methods = skip_methods
                                                                , plot_mean = plot_mean, skip = skip, plot_which = plot_which)


        if plot_which == "BR":
            data_plot = data_plot_BR
            title = 'Bayes regret'
        elif plot_which == "R":
            data_plot = data_plot_R
            title = 'Reward'
        
        if complex_x_label:
            if self.setting['order'] == "sequential":
                x_label = 'M (number of tasks)'
            else:
                x_label = 'T (number of interactions)'
        else:
            if self.setting['order'] == "sequential":
                x_label = 'M'
            else:
                x_label = 'T'

        # data_plot = data_plot.iloc[range(288),:]
        ##########################################
        line = sns.lineplot(data=data_plot
                     , x = "time", y="regret", hue="Method" # group variable
                    , ci = ci # 95, n_boot= n_boot
                    , ax = ax1
                    , n_boot = 100
                    , palette = palette
                    , linewidth = linewidth
                    , hue_order = hue_order
                    )
        if no_xtick:
            ax1.set_xticks([])

        ax1.legend().texts[0].set_text("Method")
        if w_title:
            if new_title is None:
                ax1.set_title(title, fontsize= 14)
            else:
                ax1.set_title(new_title, fontsize= 14)
        if w_x_label:
          ax1.set_xlabel(x_label, fontsize= 12)
        else:
          ax1.set_xlabel(None, fontsize= 12)
        if i == 0:
            ax1.set_ylabel(ylabel, fontsize= 14)
        else:
            ax1.set_ylabel(None, fontsize= 12)
        ax1.set(ylim=(y_min, y_max))
        #########
        handles, labels = ax1.get_legend_handles_labels()
        ax1.get_legend().remove()
    


        return data_plot, handles, labels

    ####################################################################################################
    def save(self, fig_path = "fig/", sub_folder = [], no_legend = True):
        """
        Since all results together seems quite large
        a['record'][0].keys() = (['R', 'A', 'regret', 'meta_regret', 'cum_regret', 'cum_meta_regret'])

        regret / R can almost derive anything, except for A

        The only thing is that, we may need to re-read them. Probably we need a function, to convert a "skim recorder" to a "full recorder", when we do analysis.
        """
        date = get_date()
        fig_path = fig_path + date
        if len(sub_folder) > 0:
            fig_path += "/"
            for key in sub_folder:
                fig_path += ("_" + str(key) + str(self.setting[key]))
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)

        path_settting = "_".join([str(key) + str(self.setting[key]) for key in self.setting if type(key) in [str, int] and key not in sub_folder and type(self.setting[key]) in [str, int, float]])

        print(path_settting)
        if no_legend:
            self.ax1.get_legend().remove()

        self.fig.savefig(fig_path + "/"  + path_settting + ".pdf"
                               , bbox_inches= 'tight'
                        )

    ####################################################################################################

    def save_legend(self):
        handles,labels = self.ax1.get_legend_handles_labels()
        fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(3, 2))
        axe.legend(handles, labels
                , ncol=7, numpoints=1
                  )
        axe.xaxis.set_visible(False)
        axe.yaxis.set_visible(False)
        for v in axe.spines.values():
            v.set_visible(False)
        fig.savefig("fig/legend.pdf"
                    , bbox_inches= 'tight'
                    , pad_inches = 0
                        )

