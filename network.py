from TopicModeler import Clusterer
import networkx as nx
from nameparser import HumanName
import matplotlib.pyplot as plt


class Network:

	def __init__(self):
		"""
		initializes Network class
		"""
		
		self.graph = nx.Graph()
		self.clust = self.run()
		self.nodes=list(range(6))
		self.edges=self.clust.get_edges()

	def run(self):
	    c=Clusterer(6)
	    c.load_corpus('bible')
	    c.fit()
	    c.analyze()
	    return c

	def color_by_attr(self,graph,attr):
		"""
		colors graph, currently not working
		"""
		r = lambda: random.randint(0,255)
		node_color=list()
		group_to_color=dict()
		for node in self.graph.nodes(data=True):
			a=node[1][attr]
			if a not in group_to_color.keys():
				group_to_color[a]='#%02X%02X%02X' % (r(),r(),r())
			node_color.append(group_to_color[a])
		return node_color

	def build_graph(self):
		"""
		construct the graph
		"""
		self.graph = nx.Graph()
		self.graph.add_nodes_from(self.nodes)
		e=list(set((self.edges)))
		weights=np.array([self.edges.tolist().count(t) for t in e])/len(self.edges)
		self.graph.add_weighted_edges_from([(e[i][0],e[i][1],weights[i])for i in range(len(e))])

	def show(self):
		"""
		displays graph
		"""
		nx.draw_networkx(self.graph,with_labels=False,node_size=10,node_color='blue',edge_color='green',alpha=)
		plt.show()

	def stats(self):
		print(nx.info(self.graph))
		print(f'Transitivity:  \t{nx.transitivity(self.graph)}')
		print(f'Density: \t{nx.density(self.graph)}')

widths=np.exp(weights)
widths[0]=20
fig = plt.figure(figsize=(25, 15))
plt.axis('off')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
nx.draw_networkx_nodes(G, pos,
                       nodelist=G.nodes(),
                       node_color='r',
                       node_size=750)
nx.draw_networkx_edges(G, pos,
                       edgelist=G.edges(),
                       alpha=0.5, edge_color='#5cce40', width=widths)
nx.draw_networkx_labels(G, pos, font_size=16, font_color='white')

fig.set_facecolor("#262626")
plt.savefig('results/graph2.png')
plt.show()
		

	
