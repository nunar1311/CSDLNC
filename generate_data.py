#!/usr/bin/env python3
import sys
import random
import datetime
import os
import csv
from typing import List, Dict, Any, Tuple

def generate_random_date(start_date: datetime.date, end_date: datetime.date) -> datetime.date:
    """Tạo một ngày ngẫu nhiên."""
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    return start_date + datetime.timedelta(days=random_number_of_days)

def generate_random_time() -> str:
    """Tạo thời gian ngẫu nhiên định dạng HH:MM:SS."""
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}:{second:02d}"

def generate_transaction_id() -> str:
    """Tạo Id giao dịch."""
    return f"TXN_{random.randint(10000, 999999)}"

def generate_customer_id(student_id: str, is_error: bool = False) -> str:
    """Tạo Id khách hàng. Nếu is_error là True, tạo lỗi định dạng."""
    if is_error:
        error_types = [
            lambda: f"STD{student_id}",  # Missing underscore
            lambda: f"std_{student_id}",  # Lowercase prefix
            lambda: f"STD_{student_id}_",  # Extra underscore
            lambda: f"STD-{student_id}",  # Hyphen instead of underscore
            lambda: f"STD_{random.randint(10000, 99999)}",  # Wrong student_id
            lambda: f"STD_{student_id[:2]}",  # Truncated student_id
            lambda: f"STD_",  # Missing student_id
            lambda: f"{student_id}",  # Missing prefix
            lambda: f"ST_{student_id}",  # Wrong prefix
            lambda: f" STD_{student_id}"  # Extra space
        ]
        return random.choice(error_types)()
    return f"STD_{student_id}"

def generate_product_info() -> Tuple[str, float]:
    """Tạo ngẫu nhiên thông tin sản phẩm."""
    product_categories = ["Electronics", "Clothing", "Food", "Books", "Toys"]
    category = random.choice(product_categories)
    product_id = f"{category[:3].upper()}_{random.randint(1000, 9999)}"

    # Price ranges per category
    price_ranges = {
        "Electronics": (100, 2000),
        "Clothing": (10, 200),
        "Food": (1, 50),
        "Books": (5, 100),
        "Toys": (5, 150)
    }

    min_price, max_price = price_ranges[category]
    price = round(random.uniform(min_price, max_price), 2)

    return product_id, price

def generate_payment_method(is_error: bool = False) -> str:
    """Tạo phương thức thanh toán. Nếu is_error là True, tạo lỗi định dạng."""
    payment_methods = ["CASH", "CREDIT_CARD", "DEBIT_CARD", "BANK_TRANSFER", "E_WALLET"]

    if is_error and random.random() < 0.3:
        error_methods = [
            "cash",  # Lowercase
            "Credit Card",  # Space instead of underscore
            "debit-card",  # Hyphen instead of underscore
            "BANK TRANSFER",  # Space instead of underscore
            "e_wallet",  # Lowercase
            "PAYPAL",  # Method not in regular list
            "CRYPTO",  # Method not in regular list
            "",  # Empty
            " CASH",  # Extra space
            "CASH "  # Extra space
        ]
        return random.choice(error_methods)

    return random.choice(payment_methods)

def generate_store_location(is_error: bool = False) -> str:
    """Tạo địa điểm cửa hàng. Nếu is_error là True, tạo lỗi định dạng."""
    cities = ["Hanoi", "HoChiMinh", "DaNang", "HaiPhong", "CanTho", "NhaTrang"]

    if is_error and random.random() < 0.3:
        error_locations = [
            "ha noi",  # Space and lowercase
            "HCMC",  # Abbreviation
            "Da-Nang",  # Hyphen
            "haiphong",  # Lowercase
            "Can_Tho",  # Underscore
            "NHA TRANG",  # Space and uppercase
            "",  # Empty
            " Hanoi",  # Extra space
            "HoChiMinh "  # Extra space
        ]
        return random.choice(error_locations)

    return random.choice(cities)

def generate_random_transaction(student_id: str, is_error: bool = False) -> Dict[str, Any]:
    """Tạo bản ghi giao dịch ngẫu nhiên."""
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2024, 12, 31)

    transaction_date = generate_random_date(start_date, end_date)
    transaction_time = generate_random_time()

    product_id, price = generate_product_info()
    quantity = random.randint(1, 10)

    #Cố ý tạo ra một số lỗi về số lượng nếu is_error là True
    if is_error and random.random() < 0.3:
        if random.random() < 0.5:
            quantity = str(quantity) + random.choice(["a", "b", "c"])  # Thêm chữ cái
        else:
            quantity = -quantity  # số lượng âm

    total_amount = price * quantity if isinstance(quantity, int) else 0

    # Tạo lỗi định dạng trong total_amount
    if is_error and random.random() < 0.3:
        if random.random() < 0.5:
            total_amount = str(total_amount) + random.choice(["$", "€", "£"])  # thêm kí hiệu tiền
        else:
            total_amount = ""  # total_amount trống

    transaction = {
        "transaction_id": generate_transaction_id(),
        "transaction_date": transaction_date.strftime("%Y-%m-%d"),
        "transaction_time": transaction_time,
        "customer_id": generate_customer_id(student_id, is_error),
        "product_id": product_id,
        "quantity": quantity,
        "unit_price": price,
        "total_amount": total_amount,
        "payment_method": generate_payment_method(is_error),
        "store_location": generate_store_location(is_error)
    }
    return transaction

def generate_transactions(student_id: str, num_transactions: int = 1000000) -> List[Dict[str, Any]]:
    """Tạo danh sách giao dịch với 10% lỗi cố ý"""
    transactions = []
    error_count = int(num_transactions * 0.1)  # 10% giao dịch có lỗi

    print(f"Generating {num_transactions} transactions for student ID: {student_id}")
    print(f"Including {error_count} transactions with format errors")

    for i in range(num_transactions):
        if i % (num_transactions // 10) == 0:
            print(f"Generated {i} transactions ({i/num_transactions*100:.1f}%)...")

        # Xác định xem giao dịch này có lỗi hay không
        is_error = i < error_count

        transaction = generate_random_transaction(student_id, is_error)
        transactions.append(transaction)

    # Xáo trộn danh sách giao dịch để tăng tính ngẫu nhiên
    random.shuffle(transactions)
    return transactions

def save_to_csv(transactions: List[Dict[str, Any]], student_id: str) -> str:
    """Save transactions to a CSV file and return the filename."""
    filename = f"transactions_{student_id}.csv"

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = list(transactions[0].keys())

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transactions)

    return filename

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_data.py <student_id>")
        sys.exit(1)

    student_id = sys.argv[1]

    # Kiểm tra student_id có hợp lệ không
    if not student_id.isdigit():
        print("Error: student_id must contain only digits")
        sys.exit(1)

    # Generate transactions
    transactions = generate_transactions(student_id)

    # Save to CSV
    filename = save_to_csv(transactions, student_id)

    # Get file size
    file_size = os.path.getsize(filename) / (1024 * 1024)  # Size in MB

    print(f"\nGeneration complete!")
    print(f"Data saved to: {filename}")
    print(f"File size: {file_size:.2f} MB")
    print(f"Total transactions: {len(transactions)}")

if __name__ == "__main__":
    main()