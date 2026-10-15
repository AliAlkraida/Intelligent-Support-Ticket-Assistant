import csv
import random
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(root / 'source/arabic_tickets.csv', encoding='utf-8')))
map_cat = {
 'cancel_order':'cancellations','change_order':'orders','change_shipping_address':'shipping','check_cancellation_fee':'cancellations',
 'check_invoice':'invoices','check_payment_methods':'payments','check_refund_policy':'refunds','complaint':'feedback',
 'contact_customer_service':'contact','contact_human_agent':'contact','create_account':'account','delete_account':'account',
 'delivery_options':'shipping','delivery_period':'shipping','edit_account':'account','get_invoice':'invoices',
 'get_refund':'refunds','newsletter_subscription':'newsletter','payment_issue':'payments','place_order':'orders',
 'recover_password':'account','registration_problems':'account','review':'feedback','set_up_shipping_address':'shipping',
 'switch_account':'account','track_order':'orders','track_refund':'refunds'}
# Demonstration-only English tickets; category labels are curated, reply quality is NOT gold-annotated.
examples = {
 'orders':['How can I track my order?','Can I change my order quantity?','Where is my order confirmation?','How do I place an order?'],
 'shipping':['Can I change my shipping address?','When will delivery arrive?','Can I reschedule my delivery?','What delivery options are available?'],
 'refunds':['When will my refund arrive?','How do I return an item for a refund?','Is a gift card refundable?','My card refund has not arrived.'],
 'cancellations':['Can I cancel an order after shipping?','How do I cancel my order?','Is there a cancellation fee?','I need to stop my order before shipment.'],
 'payments':['Which payment methods can I use?','My card was charged twice.','Why was my payment declined?','Can I pay using Apple Pay?'],
 'invoices':['Where can I find my invoice?','Please send me the receipt.','How do I download an invoice?','I need a tax invoice.'],
 'account':['I forgot my password.','How do I create an account?','How can I delete my account?','I cannot log in.'],
 'contact':['How do I contact a human agent?','I need customer service.','Where can I reach support?','Please connect me to an agent.'],
 'feedback':['I want to leave feedback.','How do I submit a complaint?','Where can I rate my purchase?','I want to report poor service.'],
 'newsletter':['How do I subscribe to the newsletter?','Please unsubscribe me.','Where are email subscription settings?','I want marketing emails.']}
all_rows = [{'ticket_id':f'AR{i:03}', 'text':r['text'], 'lang':'ar', 'gold_category':map_cat[r['intent']], 'source':'course_arabic_tickets', 'intent':r['intent']} for i,r in enumerate(rows,1)]
i=0
for category, texts in examples.items():
 for t in texts:
  i+=1
  all_rows.append({'ticket_id':f'EN{i:03}', 'text':t,'lang':'en','gold_category':category,'source':'authored_demonstration','intent':''})
rng=random.Random(42)
by_group=defaultdict(list)
for row in all_rows: by_group[(row['lang'],row['gold_category'])].append(row)
splits={'development':[],'validation':[],'test':[]}
for group in by_group.values():
 rng.shuffle(group)
 n=len(group)
 n_test=max(1,round(n*.2));n_val=max(1,round(n*.2))
 splits['test']+=group[:n_test]
 splits['validation']+=group[n_test:n_test+n_val]
 splits['development']+=group[n_test+n_val:]
for name, items in splits.items():
 rng.shuffle(items)
 with (root/f'{name}.csv').open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['ticket_id','text','lang','gold_category','source','intent']);writer.writeheader();writer.writerows(items)
 print(name,len(items))
