import itertools

from .roost_message import Roost
import torch.nn.functional as F
import torch
from torch_scatter import scatter_max, scatter_add
from torch_geometric.nn import MessagePassing
from .message_changed import SimpleNetwork, ResidualNetwork
from torch_geometric.utils import softmax
import torch.nn as nn
from .Hypernetworksmp import H_Net, H_Net_0


class MHAttention(nn.Module):
    """
    Multihead attention with fully connected networks used for the combination of the global features of the composition
    and the  node representations
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            heads=1,
            vector_attention=False):
        """
        Inputs
        ----------
        in_channels (int): Size of node embeddings and composition embeddings
        out_channels (int): Size of output node embedding
        heads (int): Number of attention heads
        vector_attention (bool): If set to true vectorized attention coefficients are used
        """
        super(MHAttention, self).__init__()
        self.heads = heads
        self.out_channels = out_channels
        if(vector_attention):
            self.MH_A = MultiHeadNetwork(
                2 * in_channels,
                out_channels,
                in_channels,
                heads,
                view=False)
        else:
            self.MH_A = MultiHeadNetwork(
                2 * in_channels, 1, in_channels, heads, view=False)
        self.MH_M = MultiHeadNetwork(
            in_channels, out_channels, in_channels, heads)

    def forward(self, fea, cry_fea, index, size=None):
        """ forward pass """
        size = index[-1].item() + 1 if size is None else size
        m = self.MH_M(fea)
        # concatenate atomic and global featues for the corresponding sytem
        fea = torch.stack([fea, cry_fea[index]])
        # switch axis to get correct reshaping in Multiheadnetworks
        fea = fea.transpose(1, 0)
        alpha = self.MH_A(fea)
        alpha = softmax(alpha, index, None, size)
        out = scatter_add((alpha * m).view(-1, self.heads * \
                          self.out_channels), index, dim=0, dim_size=size)
        return out


class MultiHeadNetwork(nn.Module):
    """
    nb_heads parallel feed forward networks to be used in MHAttention
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_layer_dim,
            nb_heads,
            view=True):
        """
        Inputs
        ----------
        input_dim (int): Input size
        output_dim (int): Outputsize
        hidden_layer_dim (int): Hidden layer dimension
        nb_heads (int): Number of attention heads/fully  connected networks
        view (bool): Set to False if fea tensor is not contiguous in memory
        """
        super(MultiHeadNetwork, self).__init__()

        self.input_dim = input_dim
        self.nb_heads = nb_heads
        self.output_dim = output_dim
        self.fc_in = nn.Conv1d(in_channels=input_dim * nb_heads,
            out_channels= hidden_layer_dim * nb_heads,
            kernel_size=1,
            groups=nb_heads)
        self.acts = nn.LeakyReLU()
        self.fc_out = nn.Conv1d(
            in_channels=hidden_layer_dim * nb_heads,
            out_channels=output_dim * nb_heads,
            kernel_size=1,
            groups=nb_heads)
        self.view = view

    def forward(self, fea):
        if self.view:
            fea = self.acts(self.fc_in(fea.view(-1, self.input_dim, 1).repeat(1, self.nb_heads, 1)))
        else:
            fea = self.acts(self.fc_in(fea.reshape(-1, self.input_dim, 1).repeat(1, self.nb_heads, 1)))

        return self.fc_out(fea).view(-1, self.nb_heads, self.output_dim)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class GATConvEdges(nn.Module):
    """ graph attentional operator for edges combines node and edge information
        and updates the edge embedding through a multihead attention mechanism
    Args:
        in_channels (int): Size of node embedding.
        out_channels (int): Size of output embedding.
        nbr_channels (int): Size of edge embeddings
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        vector_attention (bool, optional): If set to :obj:`False`, the attention coefficients will be scalar else they
        will be vectors (default: :obj:`False`)
        first (bool, optional): Ignore (default: :obj:`True`)
        no_hyper (bool, optional): If set to False will use hypernetworks for pooling_NN (default :obj'False')
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            nbr_channels,
            heads=1,
            concat=True,
            negative_slope=0.2,
            dropout=0,
            bias=True,
            vector_attention=False,
            first=False,
            no_hyper=True,
            **kwargs):
        super(GATConvEdges, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nbr_channels = nbr_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.vector_attention = vector_attention

        if(vector_attention):
            self.MH_A = MultiHeadNetwork(2 * in_channels + nbr_channels,
                                         out_channels,
                                         int((2 * in_channels + nbr_channels) / 1.5),
                                         heads)
        else:
            self.MH_A = MultiHeadNetwork(2 * in_channels + nbr_channels,
                                         1,
                                         int((2 * in_channels +  nbr_channels) / 1.5),
                                         heads)

        self.MH_M = MultiHeadNetwork(2 * in_channels + nbr_channels,
                                     out_channels,
                                     int((2 * in_channels + nbr_channels) / 1.5),
                                     heads)

        if no_hyper:
            self.Pooling_NN = SimpleNetwork(out_channels,
                                            out_channels,
                                            [out_channels])
        elif first:
            self.Pooling_NN = H_Net_0(
                out_channels,
                3,
                out_channels,
                out_channels,
                2,
                out_channels,
                out_channels)
        else:
            self.Pooling_NN = H_Net(
                out_channels,
                3,
                out_channels,
                out_channels,
                2,
                out_channels,
                out_channels)
        self.first = first
        self.no_hyper = no_hyper

    def forward(self, x, edge_index, edge_attr, x_0, size=None):
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]
        m = torch.cat([x_i, edge_attr, x_j], dim=-1)
        alpha = self.MH_A(m)
        m = self.MH_M(m)
        alpha = alpha.exp()

        if(not self.vector_attention):
            alpha = alpha / alpha.sum(dim=1).view(-1, 1, 1)
        else:
            alpha = alpha / alpha.sum(dim=1).view(-1, 1, self.out_channels)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        aggr_out = m.view(-1, self.heads, self.out_channels) * alpha
        aggr_out = aggr_out.mean(dim=1)
        if self.no_hyper:
            aggr_out = self.Pooling_NN(edge_attr)
        elif self.first:
            aggr_out = self.Pooling_NN(edge_attr, aggr_out)
        else:
            aggr_out = self.Pooling_NN(x_0, edge_attr, aggr_out)
        return aggr_out


class GATConvNodes(MessagePassing):
    """Graph attentional operator adapted from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper
    Args:
        in_channels (int): Size of node embedding.
        out_channels (int): Size of output embedding.
        nbr_channels (int): Size of edge embeddings
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        final (bool, optional):  :obj:`False`, Should be set to false for the last message passing layer
            (default: :obj:`False`)
        vector_attention (bool, optional): If set to :obj:`False`, the attention coefficients will be scalar else they
        will be vectors (default: :obj:`False`)
        first (bool, optional): Ignore (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            nbr_channels,
            heads=1,
            concat=False,
            negative_slope=0.2,
            dropout=0,
            bias=True,
            final=False,
            vector_attention=False,
            first=False,
            **kwargs):
        super(GATConvNodes, self).__init__(aggr='add', **kwargs)
        self.node_dim = 0 #propagation axis
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nbr_channels = nbr_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.final = final
        self.first = first
        if vector_attention:
            self.MH_A = MultiHeadNetwork(2 * in_channels + nbr_channels,
                                         out_channels,
                                         int((2 * in_channels + nbr_channels) / 1.5),
                                         heads)
        else:
            self.MH_A = MultiHeadNetwork(2 * in_channels + nbr_channels,
                                         1,
                                         int((2 * in_channels + nbr_channels) / 1.5),
                                         heads)
        self.MH_M = MultiHeadNetwork(2 * in_channels + nbr_channels,
                                     out_channels,
                                     int((2 * in_channels + nbr_channels) / 1.5),
                                     heads)
        if not final and first:
            self.Pooling_NN = H_Net_0(out_channels, 3, out_channels, out_channels,
                                      2, out_channels, out_channels)
        elif not final:
            self.Pooling_NN = H_Net(out_channels, 3, out_channels, out_channels,
                                      2, out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, x_0, size=None):
        if torch.is_tensor(x):
            pass
        else:
            x = (None if x[0] is None else x[0],
                 None if x[1] is None else x[1])
        return self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            x_0=x_0)

    def message(self, x_i, x_j, edge_attr, edge_index_i):
        m = torch.cat([x_i, edge_attr, x_j], dim=-1)
        alpha = self.MH_A(m)
        m = self.MH_M(m)
        alpha = softmax(alpha, edge_index_i) #, size_i)
        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return m * alpha

    def update(self, aggr_out, x_0, x):
        aggr_out = aggr_out.mean(dim=1)
        if not self.final and self.first:
            return self.Pooling_NN(x, aggr_out)
        elif not self.final:
            return self.Pooling_NN(x_0, x, aggr_out)
        else:
            return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class CGAtNet(nn.Module):
    """
    Create a neural network for predicting total material properties.

    The CGatNet is comprised of a fully connected output network,
    message passing graph layers used on a crystal graph and a composition based roost model
    (see https://github.com/CompRhys/roost and https://doi.org/10.1038/s41467-020-19964-7 for the roost model) that is
    used in the final pooling step.

    The message passing layers are used to determine an embedding for the whole material that is used in the
    fully connected network. Critically the graphs are used to
    represent (crystalline) materials in a volume agnostic manner.
    The model is also agnostic to changes in the structure that do not cause a change in
    neighborlist.
    """

    def __init__(
            self,
            orig_elem_fea_len,
            elem_fea_len,
            n_graph,
            nbr_embedding_size=128,
            neighbor_number=12,
            mean_pooling=True,
            rezero=False,
            msg_heads=3,
            update_edges=False,
            vector_attention=False,
            global_vector_attention=False,
            n_graph_roost=3,
            no_hyper=True):
        """
    Args:
        orig_elem_fea_len (int): size of pretrained species embedding.
        elem_fea_len (int): Size of species embedding used during message passing
        n_graph (int): Number of message passing steps in CGAT modell
        nbr_embedding_size (int): Size of edge embeddings
        neighbor_number (int): Number of neighbors considered during each message passing step
        mean_pooling (int): If set to False the material embeddings returned by the pooling layer following the message
            passing will be concatenated instead of averaged (default: obj: True)
        msg_heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`3`)
        update_edges (bool): If set to True edge embeddings will be updated (default: :obj:`False`)
        vector_attention (bool, optional): If set to :obj:`False`, the attention coefficients during the message
            passing phase will be scalar else they will be vectors (default: :obj:`False`)
        global_vector_attention (bool, optional): If set to :obj:`False`, the attention coefficients of the pooling
            layer after the message passing phase will be scalar else they will be vectors (default: :obj:`False`)
        n_graph_roost (int): Number of message passing steps in the roost model
        no_hyper (bool, optional): If set to :obj:`False`, hypernetworks will be used in the message passing of the edges
        **kwargs (optional): Additional arguments of :class:`torch_geometric.nn.conv.MessagePassing`.
    """
        super(CGAtNet, self).__init__()
        self.mean_pooling = mean_pooling
        self.update_edges = update_edges
        # apply linear transform to the input to get a trainable embedding
        self.embedding = nn.Linear(orig_elem_fea_len, elem_fea_len, bias=False)
        self.nbr_embedding = nn.Embedding(
            num_embeddings=neighbor_number + 1,
            embedding_dim=nbr_embedding_size)
        self.no_hyper = no_hyper
        # create a list of Message passing layers
        msg_heads = msg_heads
        if not self.update_edges:
            self.graphs = nn.ModuleList(
                [
                    GATConvNodes(elem_fea_len,
                                 nbr_embedding_size,
                                 msg_heads,
                                 concat=True,
                                 vector_attention=vector_attention,
                                 first=True)
                ])
            self.graphs.extend(
                [
                    GATConvNodes(elem_fea_len,
                                 nbr_embedding_size,
                                 msg_heads,
                                 concat=True,
                                 vector_attention=vector_attention)
                    for i in range(
                        n_graph - 1)
                ]
            )

        elif no_hyper: #no hyper networks for edge updates
            self.graphs = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            'Node': GATConvNodes(elem_fea_len,
                                                 elem_fea_len,
                                                 nbr_embedding_size,
                                                 msg_heads,
                                                 concat=True,
                                                 vector_attention=vector_attention,
                                                 first=True),
                            'Edge': GATConvEdges(elem_fea_len,
                                                 nbr_embedding_size,
                                                 nbr_embedding_size,
                                                 msg_heads,
                                                 concat=True,
                                                 vector_attention=vector_attention,
                                                 first=True)
                        }
                    )
                ]
            )
            self.graphs.extend(
                [
                    nn.ModuleDict(
                        {
                            'Node': GATConvNodes(elem_fea_len,
                                                 elem_fea_len,
                                                 nbr_embedding_size,
                                                 msg_heads,
                                                 concat=True,
                                                 vector_attention=vector_attention),

                            'Edge': GATConvEdges(elem_fea_len,
                                                 nbr_embedding_size,
                                                 nbr_embedding_size,
                                                 msg_heads,
                                                 concat=True,
                                                 vector_attention=vector_attention)
                        }
                    ) for i in range(n_graph - 1)])
        else:
            self.graphs = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            'Node': GATConvNodes(elem_fea_len,
                                                 elem_fea_len,
                                                 nbr_embedding_size,
                                                 msg_heads,
                                                 concat=True,
                                                 vector_attention=vector_attention,
                                                 first=True),

                            'Edge': GATConvEdges(elem_fea_len,
                                                 nbr_embedding_size,
                                                 nbr_embedding_size,
                                                 msg_heads,
                                                 concat=True,
                                                 vector_attention=vector_attention,
                                                 first=True,
                                                 no_hyper=False)})])
            self.graphs.extend(
                [
                    nn.ModuleDict(
                        {
                            'Node': GATConvNodes(elem_fea_len,
                                                 elem_fea_len,
                                                 nbr_embedding_size,
                                                 msg_heads,
                                                 concat=True,
                                                 vector_attention=vector_attention),

                            'Edge': GATConvEdges(elem_fea_len,
                                                 nbr_embedding_size,
                                                 nbr_embedding_size,
                                                 msg_heads,
                                                 concat=True,
                                                 vector_attention=vector_attention,
                                                 no_hyper=False)
                        }
                                )
                for i in range(n_graph - 1)]
            )

        # Add a roost model for a composition based pooling
        self.roost = Roost(orig_elem_fea_len, elem_fea_len, n_graph_roost)

        # define a global pooling function for materials
        mat_heads = msg_heads
        self.cry_pool = MHAttention(in_channels=elem_fea_len,
                                    out_channels=elem_fea_len,
                                    heads=mat_heads,
                                    vector_attention=global_vector_attention)

        # define an output neural network
        self.msg_heads = msg_heads
        self.elem_fea_len = elem_fea_len
        out_hidden = [1024, 1024, 512, 512, 256, 256, 128]

        if mean_pooling:
            self.output_nn = ResidualNetwork(elem_fea_len,
                                             2,
                                             out_hidden,
                                             if_rezero=rezero)
        else:
            self.output_nn = ResidualNetwork(elem_fea_len * msg_heads,
                                             2,
                                             out_hidden,
                                             if_rezero=rezero)


    def forward(self, batch, roost):
        """
        Forward pass

        Parameters
        ----------
        batch:
            Batch of torch_geometric graph objects
        roost:
        ----------
        orig_elem_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
            Atom features of each of the N elems in the batch
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the elem each of the M bonds correspond to
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of of the neighbours of the M bonds connect to
        elem_bond_idx: list of torch.LongTensor of length C
            Mapping from the bond idx to elem idx
        crystal_elem_idx: list of torch.LongTensor of length C
            Mapping from the elem idx to crystal idx

        Returns
        -------
        out: nn.Variable shape (C,)
            Atom hidden features after message passing
        """
        edge_index = batch.edge_index
        crystal_elem_idx = batch.batch
        size = (batch.num_nodes, batch.num_nodes)
        edge_attr = self.nbr_embedding(batch.edge_attr)
        elem_fea = self.embedding(batch.x)
        elem_fea_0 = elem_fea.clone()

        if self.update_edges:
            edge_attr_0 = edge_attr.clone()
        if(not self.update_edges):
            for graph_func in self.graphs:
                elem_fea = elem_fea + \
                    graph_func(elem_fea, edge_index, edge_attr, elem_fea_0, size)
        else:
            for graph_func in self.graphs:
                node_update = graph_func['Node'](
                    elem_fea, edge_index, edge_attr, elem_fea_0, size)
                edge_attr = edge_attr + \
                    graph_func['Edge'](elem_fea, edge_index, edge_attr, edge_attr_0, size)
                elem_fea = elem_fea + node_update

        crys_fea = self.roost(*roost)
        crys_fea = self.cry_pool(elem_fea, crys_fea, crystal_elem_idx)

        if(self.mean_pooling):
            crys_fea = crys_fea.view(-1, self.msg_heads, self.elem_fea_len)
            crys_fea = torch.mean(crys_fea, dim=1)
            crys_fea = self.output_nn(crys_fea)
        else:
            crys_fea = self.output_nn(crys_fea)
        return crys_fea

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

    def get_output_parameters(self):
        return self.output_nn.parameters()

    def get_hidden_parameters(self):
        return itertools.chain(self.embedding.parameters(),
                               self.nbr_embedding.parameters(),
                               self.graphs.parameters(),
                               self.roost.parameters(),
                               self.cry_pool.parameters())
