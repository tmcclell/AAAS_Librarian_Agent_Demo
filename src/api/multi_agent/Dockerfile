# syntax=docker/dockerfile:1

FROM python:3.11-slim

# 1. Update and install prerequisites (curl, gnupg, etc.)
RUN apt-get update -y && \
    apt-get install -y curl gnupg apt-transport-https ca-certificates

# 2. Add the Microsoft GPG key and repo.
#    For Debian 11 (bullseye):
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/11/prod.list | tee /etc/apt/sources.list.d/msprod.list

# 3. Update again and install the MS ODBC driver + unixODBC dev libs
RUN apt-get update -y && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18 unixodbc-dev

# 4. (Optional) You may also want tools like these:
# RUN apt-get install -y libssl1.1 libgssapi-krb5-2  # or other libraries, if needed

# 5. Create a working directory
WORKDIR /code

# 6. Copy your application code
COPY api/sql_agent/ . 
COPY utility_functions . 
COPY . .

# 7. Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --upgrade -r requirements.txt

# 8. Expose the port
EXPOSE 3100

# 9. Start the app
CMD ["gunicorn", "sql_agent:app"]
