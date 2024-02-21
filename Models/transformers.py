import numpy as np
import torch
import torch.nn as nn
import math
class AddNorm(nn.Module):
    def __init__(self, norm_shape):
        super().__init__()
        self.activation = nn.ReLU()
        self.normalization = nn.LayerNorm(norm_shape)
    def forward(self, x,y):
        return self.normalization(y + x)# + x




class InnerSelfAttention(nn.Module):
    def __init__(self, hidden_size_actual, n_heads, max_n_codes, device):
        super().__init__()
        hidden_size = int(int(hidden_size_actual / n_heads) * n_heads)
        self.hidden_states = hidden_size
        self.device = device
        self.num_attention_heads = n_heads
        self.max_n_codes = max_n_codes
        self.attention_head_size = int(hidden_size / n_heads)

        self.all_head_size = hidden_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = nn.Tanh()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def get_query(self, input):
        return self.query(input)

    def get_key(self, input):
        return self.key(input)

    def create_padding_matrix(self, padding_vector, max_number):
        padding_matrix = []
        un_used_padding = []
        for index in range(0, padding_vector.size()[0]):
            padd_vectors = (padding_vector[index, :] == 0).nonzero(as_tuple=True)
            try:
                padd = padd_vectors[0][padd_vectors[0].sort()[1]][0].item()
            except:
                padd = max_number
            p_array = np.full((max_number, max_number), 1)
            up_array = np.full((max_number, max_number), 0)
            p_array[0:padd, 0:padd] = 0
            #p_array[padd:, padd:] = 0
            up_array[padd:, padd:] = 1
            padding_matrix.append(p_array)
            un_used_padding.append(up_array)
        padding_matrix = np.array(padding_matrix)
        padding_matrix_heads = np.expand_dims(padding_matrix, axis=1)
        un_used_padding = np.array(un_used_padding)
        un_used_padding_heads = np.expand_dims(un_used_padding, axis=1)

        for i in range(1, self.num_attention_heads):
            padding_matrix_heads = np.concatenate((padding_matrix_heads, np.expand_dims(padding_matrix, axis=1)),
                                                  axis=1)
            un_used_padding_heads = np.concatenate((un_used_padding_heads, np.expand_dims(padding_matrix, axis=1)),
                                                  axis=1)

        return padding_matrix_heads, un_used_padding_heads

    def forward(self, hidden_states, padding_vector, max_n_codes, interpretabity, iter):
        mixed_query_layer = self.query(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(hidden_states)  # [Batch_size x Seq_length x Hidden_size]

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        #print('shapeofhead', value_layer.size())
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
                                                                         -2))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        padding_matrix, un_used_padding_matrix = self.create_padding_matrix(padding_vector, max_n_codes)
        attention_masked = attention_scores.to(self.device).masked_fill(torch.tensor(padding_matrix).type(torch.bool).to(self.device), float('-inf'))
        attention_probs = nn.Softmax(dim = -1)(attention_masked)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_probes_mask = attention_probs.masked_fill(torch.tensor(padding_matrix).type(torch.bool).to(self.device), float(0))
        attention_values = torch.sum(attention_probes_mask, dim=-2) # [Batch_size x Num_of_heads x Seq_length]
        context_layer = torch.matmul(attention_probes_mask.to(self.device),
                             value_layer.to(self.device))  # [Batch_size x Num_of_heads x Seq_length x Head_size]


        return attention_values, key_layer, query_layer, context_layer, attention_scores

class SimpleSelfAtt(nn.Module):
    def __init__(self, hidden_size_actual, n_heads, max_n_codes, device):
        super().__init__()
        hidden_size = int(int(hidden_size_actual / n_heads) * n_heads)
        self.hidden_states = hidden_size
        self.device = device
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.max_n_codes = max_n_codes
        self.softmax = nn.Softmax(dim=-1)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.value_t_1 = nn.Linear(hidden_size, self.all_head_size)
        self.concat = nn.Linear(2 * self.attention_head_size, 1)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
        self.add_norm = AddNorm(self.attention_head_size)

    def create_padding_matrix(self, padding_vector, max_number):
        padding_matrix = []
        for index in range(0, padding_vector.size()[0]):
            padd_vectors = (padding_vector[index, :] == 0).nonzero(as_tuple=True)
            try:
                padd = padd_vectors[0][padd_vectors[0].sort()[1]][0].item()
            except:
                padd = max_number
            p_array = np.full(max_number, 1)
            p_array[0:padd] = 0
            padding_matrix.append(p_array)
        padding_matrix = np.array(padding_matrix)
        padding_matrix_heads = np.expand_dims(padding_matrix, axis=1)

        for i in range(1, self.num_attention_heads):
            padding_matrix_heads = np.concatenate((padding_matrix_heads, np.expand_dims(padding_matrix, axis=1)),
                                                  axis=1)

        return padding_matrix_heads


    def forward(self, transformer_t_1, transformer_t, padding_t, max_n_codes):
        #B,heads,nodes,dim
        padding_matrix = self.create_padding_matrix(padding_t, max_n_codes)

        all_nodes_t_1 = transformer_t_1.sum(dim = 2) #B,heads,dim
        all_nodes_t_1 = all_nodes_t_1.unsqueeze(-1).repeat(1,1,1,transformer_t_1.size()[2]).permute(0,1,3,2) #B,heads,nodes,dim

        cat_times = self.concat(torch.cat((all_nodes_t_1, transformer_t), dim = -1)).squeeze(-1)

        attention_masked = cat_times.masked_fill(torch.tensor(padding_matrix).type(torch.bool), float('-inf'))

        softmax = self.softmax(attention_masked)

        attention_masked = softmax.masked_fill(torch.tensor(padding_matrix).type(torch.bool), float(0))


        return (transformer_t*attention_masked.unsqueeze(-1)) + transformer_t


class InterSelfAttention(nn.Module):
    def __init__(self, hidden_size_actual, n_heads, max_n_codes, device):
        super().__init__()
        hidden_size = int(int(hidden_size_actual / n_heads) * n_heads)
        self.hidden_states = hidden_size
        self.device = device
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.max_n_codes = max_n_codes
        self.softmax = nn.Softmax(dim=-1)
        self.value = nn.Linear(hidden_size, self.all_head_size).to(self.device)
        self.value_t_1 = nn.Linear(hidden_size, self.all_head_size).to(self.device)
        self.query = nn.Linear(hidden_size, self.all_head_size).to(self.device)
        self.key = nn.Linear(hidden_size, self.all_head_size).to(self.device)

        self.concat = nn.Linear(2*self.attention_head_size, self.attention_head_size)
        self.activation = nn.Tanh()
        self.add_norm = AddNorm(self.attention_head_size)
        self.concat = self.concat.to(self.device)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def create_padding_matrix(self, padding_vector_q, padding_vector_k, max_number):
        padding_matrix = []
        un_used_padding = []

        for index in range(0, len(padding_vector_q)):
            padd_vectors = (padding_vector_q[index, :] == 0).nonzero(as_tuple=True)
            try:
                padd_q = padd_vectors[0][padd_vectors[0].sort()[1]][0].item()
            except:
                padd_q = max_number

            padd_vectors = (padding_vector_k[index, :] == 0).nonzero(as_tuple=True)
            try:
                padd_k = padd_vectors[0][padd_vectors[0].sort()[1]][0].item()
            except:
                padd_k = max_number
            #if index ==0:
            #    print('current',padd_k,'last',padd_q)
            #if index == 0:
            #    print(padd_k, padd_q)
            #    print(padding_vector_k[index, :], padding_vector_q[index, :])
            up_array = np.full((max_number, max_number), 0)
            p_array = np.full((max_number, max_number), 1)
            p_array[0:padd_q, 0:padd_k] = 0
            #p_array[padd_q:, padd_k:] = 0
            up_array[padd_q:, padd_k:] = 1
            padding_matrix.append(p_array)
            un_used_padding.append(up_array)
        padding_matrix = np.array(padding_matrix)
        padding_matrix_heads = np.expand_dims(padding_matrix, axis=1)
        un_used_padding = np.array(un_used_padding)
        un_used_padding_heads = np.expand_dims(un_used_padding, axis=1)
        for i in range(1, self.num_attention_heads):
            padding_matrix_heads = np.concatenate((padding_matrix_heads, np.expand_dims(padding_matrix, axis=1)),
                                                  axis=1)
            un_used_padding_heads = np.concatenate((un_used_padding_heads, np.expand_dims(padding_matrix, axis=1)),
                                                   axis=1)
        return padding_matrix_heads, un_used_padding_heads, padd_q, padd_k

    def forward(self, padding_vector_q, padding_vector_k, query_t_1, key_t, hidden_states,hidden_states_t_1, hidden_from_last, hidden_current, max_n_codes, interpretabity, iter,time=0):
        #print(self.device)
        #mixed_value_layer = self.value(hidden_current.permute(0,2,1,3).flatten(start_dim = 2))
        self.value_t_1 = self.value_t_1.to(self.device)
        self.value = self.value.to(self.device)
        self.query = self.query.to(self.device)
        self.key = self.key.to(self.device)
        self.concat = self.concat.to(self.device)
        mixed_value_t_1_layer = self.value_t_1(hidden_from_last.permute(0,2,1,3).flatten(start_dim = 2))
        mixed_value_t_layer = self.value(hidden_current.permute(0,2,1,3).flatten(start_dim = 2))

        mixed_query_layer = self.query(hidden_from_last.permute(0,2,1,3).flatten(start_dim = 2))
        mixed_key_layer = self.key(hidden_current.permute(0,2,1,3).flatten(start_dim = 2))

        value_layer_t_1 = self.transpose_for_scores(
            mixed_value_t_1_layer)
        value_layer_t = self.transpose_for_scores(
            mixed_value_t_layer)
        query_layer = self.transpose_for_scores(
            mixed_query_layer)
        key_layer = self.transpose_for_scores(
            mixed_key_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
                                                                         -2))

        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores.permute(0, 1, 3, 2)

        padding_matrix, un_used_padding_matrix, padd_q, padd_k = self.create_padding_matrix(padding_vector_q, padding_vector_k, max_n_codes)

        attention_masked = attention_scores.to(self.device).masked_fill(torch.tensor(padding_matrix).type(torch.bool).to(self.device),
                                                        float('-inf'))
        attention_masked = nn.Softmax(dim = -1)(attention_masked)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]

        attention_values = attention_masked.masked_fill(
            torch.tensor(padding_matrix).type(torch.bool).to(self.device),
                                                           float(0))

        context_layer = torch.matmul(attention_values.to(self.device), value_layer_t_1.to(self.device))

        context_layer2 = self.concat(torch.cat((context_layer, value_layer_t), dim=-1))#+ value_layer
        return attention_values, attention_scores, context_layer2, context_layer #+ hidden_current
