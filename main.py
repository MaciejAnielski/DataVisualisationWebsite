# Import objects.

from polivis import CreateDirectory
from polivis import UKVotingIntentionProcessor
from polivis import USPresidentialApprovalProcessor

# Set up directory structure.

CreateDirectory()

# Create UK voting intention graphs.

ukvi = UKVotingIntentionProcessor()
ukvi.plot_voting_intention(election_year = ['next', '2024'])

# Create US presidential approval graphs.
 
uspi = USPresidentialApprovalProcessor()

uspi.plot_approval_2017_present()
uspi.plot_approval_current()