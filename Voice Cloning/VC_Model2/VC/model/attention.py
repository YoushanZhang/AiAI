
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F



class BahdanauAttention(nn.Module):
	def __init__(self, dim):
		super(BahdanauAttention, self).__init__()
		self.query_layer = nn.Linear(dim, dim, bias=False)
		self.tanh = nn.Tanh()
		self.v = nn.Linear(dim, 1, bias=False)

	"""
		Args:
			query: (batch, 1, dim) or (batch, dim)
			processed_memory: (batch, max_time, dim)
	"""
	def forward(self, query, processed_memory):
		if query.dim() == 2:
			query = query.unsqueeze(1) # insert time-axis for broadcasting

		processed_query = self.query_layer(query) # (batch, 1, dim)

		alignment = self.v(self.tanh(processed_query + processed_memory)) # (batch, max_time, 1)

		return alignment.squeeze(-1) # (batch, max_time)


"""
	Get mask tensor from list of length

	Args:
		memory: (batch, max_time, dim)
		memory_lengths: array like
"""
def get_mask_from_lengths(memory, memory_lengths):
	mask = memory.data.new(memory.size(0), memory.size(1)).byte().zero_()
	for idx, l in enumerate(memory_lengths):
		mask[idx][:l] = 1
	return ~mask



class AttentionWrapper(nn.Module):

	def __init__(self, 
				 rnn_cell, 
				 attention_mechanism,
				 score_mask_value=-float("inf")):

		super(AttentionWrapper, self).__init__()

		self.rnn_cell = rnn_cell
		self.attention_mechanism = attention_mechanism
		self.score_mask_value = score_mask_value

	def forward(self,
				query, 
				attention, 
				cell_state, 
				memory,
				processed_memory=None, mask=None, memory_lengths=None):
		
		if processed_memory is None:
			processed_memory = memory
		if memory_lengths is not None and mask is None:
			mask = get_mask_from_lengths(memory, memory_lengths)

		cell_input = torch.cat((query, attention), -1) # Concat input query and previous attention context

		cell_output = self.rnn_cell(cell_input, cell_state) # Feed it to RNN

		alignment = self.attention_mechanism(cell_output, processed_memory) # Alignment: (batch, max_time)

		if mask is not None:
			mask = mask.view(query.size(0), -1)
			alignment.data.masked_fill_(mask, self.score_mask_value)

		alignment = F.softmax(alignment, dim=1) # Normalize attention weight

		attention = torch.bmm(alignment.unsqueeze(1), memory) # Attention context vector: (batch, 1, dim)

		attention = attention.squeeze(1) # (batch, dim)

		return cell_output, attention, alignment
