system_messages = '''

You are a report generator and will help me generate a report based on instructions provided.

'''

def generate_prompt(content ):
  prompt = f"""

I want to summarize the Sales Force Incidents that were opened. 
Please look at each of the C360 CS Task ID and generate a report with a separate section for each C360 CS Task ID 
Each C360 CS Task ID section should contain:
- 2 Sections.  
- The Summary Section, where each column and it's values should be shown, one row per entry.
- Next Please create a Recommendation / Resolution / Next Step section, with sub headers for different categories of recommendations.
Please make sure to clearly highlight:
 - who should be involved or who should take what steps if the information is available. 
 - Diagnostic and Monitoring:  if applicable
 - Responsible Party: if applicable
 - Steps to Take: if applicable
 - Customer Enablement: section with Action Required: Responsible Party: and Steps to Take: sub-sections if available
 - Communication and Follow-Up: section with Action Required: Responsible Party: and Steps to Take: sub-sections if available
 - Case Management: section with Action Required: Responsible Party: and Steps to Take: sub-sections if available
 - Technical Support: section with Action Required: Responsible Party: and Steps to Take: sub-sections if available
 - Next Steps: section with Immediate Action: Responsible Party: Long-Term Action: sub-sections if applicable
 - Any other subsection that might make for a good report to present to management



The content that you should analyze is {content}
----




  """
  return prompt