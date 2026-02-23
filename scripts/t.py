import snowflake.connector
totp = input("Enter TOTP code: ")
conn = snowflake.connector.connect(
    account="SFEDU02-DCB73175",  # paste exact value from URL
    user="COBRA",
    password="wilds!5!COBRARB",
    passcode=totp.strip(),
    authenticator="username_password_mfa"
)