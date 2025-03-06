# Import objects.

from polivis import CreateDirectory
from polivis import UKVotingIntentionProcessor
from polivis import USPresidentialApprovalProcessor

# Set up directory structure.

CreateDirectory()

# Create UK voting intention graphs.

ukvi = UKVotingIntentionProcessor()

ukvi_data = ukvi.clean_wiki_tables()

ukvi.plot_voting_intention(ukvi_data = ukvi_data, election_year = ['next'])
ukvi.plot_voting_intention(ukvi_data = ukvi_data, election_year = ['next', '2024', '2019', '2017', '2015', '2010'])

# Create US presidential approval graphs.
 
uspi = USPresidentialApprovalProcessor()

uspi.plot_approval_2017_present()
uspi.plot_approval_current()